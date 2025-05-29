import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.models import BooleanModel, LSAModel
from ui.sidebar import load_sidebar
from ui.utils import compute_metrics, load_documents, load_model

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔍",
    layout="wide",
)


def plot_term_frequency(model: BooleanModel) -> None:
    st.subheader("Term Frequency")

    # Get most common terms
    term_doc_counts = {term: len(docs) for term, docs in model.inverted_index.items()}
    top_k = st.number_input("Number of top terms", 1, len(term_doc_counts), 20)
    most_common = sorted(term_doc_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]  # fmt: skip

    # Visualization
    df = pd.DataFrame(most_common, columns=["Term", "Document Count"])
    st.dataframe(df)

    fig = px.bar(
        df.head(min(300, len(df))),
        x="Term",
        y="Document Count",
        title=f"Top {min(300, len(df))} Most Frequent Terms",
    )
    st.plotly_chart(fig)


def plot_top_terms(model: LSAModel) -> None:
    st.subheader("Top terms per component")

    # Get top terms
    col1, col2 = st.columns(2)
    with col1:
        num_components = st.number_input("Select component", 0, model.n_components - 1)
    with col2:
        num_terms = st.number_input("Number of terms", 1, len(model.vocabulary), 20)

    top_terms = model.get_top_terms(num_components, num_terms)
    df = pd.DataFrame(
        {
            "Term": top_terms[0],
            "Weight": top_terms[1],
            "Magnitude": np.abs(top_terms[1]),
        }
    )
    st.dataframe(df)

    # Plot top terms
    k_terms = min(300, len(df))
    magnitude_fig = px.bar(
        df.head(k_terms),
        x="Term",
        y="Magnitude",
        title=f"Top {k_terms} terms for component {num_components} (Magnitude)",
    )
    weight_fig = px.bar(
        df.head(k_terms),
        x="Term",
        y="Weight",
        title=f"Top {k_terms} terms for component {num_components} (Weight)",
    )
    st.plotly_chart(magnitude_fig)
    st.plotly_chart(weight_fig)


def main() -> None:
    st.title("🔍 Text Retrieval System")

    # Initialize session state variables
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "model" not in st.session_state:
        st.session_state.model = None

    # Main content area
    tab_search, tab_stats, tab_docs = st.tabs(["Search", "Statistics", "Documents"])

    # Sidebar
    data_params, model_params, load_button = load_sidebar()
    source_type = data_params.pop("source_type")
    model_type = model_params.pop("model_type")

    # Load data and initialize model
    if load_button:
        with st.spinner("Loading documents and initializing model..."):
            try:
                documents = load_documents(source_type, **data_params)

                if not documents:
                    st.error("No documents loaded. Please check and try again.")
                else:
                    model = load_model(model_type, **model_params)
                    model.fit(documents)

                    st.session_state.documents = documents
                    st.session_state.model_type = model_type
                    st.session_state.model = model
                    st.success(
                        f"Successfully loaded {len(documents)} documents and initialized {model_type} model!"
                    )
            except Exception as e:
                st.error(f"Error loading data: {e}")

    with tab_search:
        if st.session_state.model is not None:
            st.header("Search Documents")
            query = st.text_input("Enter search query")

            # Search interface based on model type
            search_params = {}
            if st.session_state.model_type != "Boolean":
                if st.session_state.model_type == "Combined":
                    search_params["boolean_query"] = st.text_input("Enter Boolean query (optional)")  # fmt: skip
                    search_params["search_mode"] = st.radio("Search mode", ["boolean_first", "lsa_first"])  # fmt: skip

                search_params["threshold"] = st.slider("Score threshold", 0.0, 1.0, 0.05)  # fmt: skip

                use_top_k = st.checkbox("Use top-k results", value=True)
                top_k = st.number_input(
                    "Number of results",
                    min_value=1,
                    max_value=len(st.session_state.documents),
                    value=10,
                    disabled=not use_top_k,
                )
                if use_top_k:
                    search_params["top_k"] = top_k

            if st.button("Search") and query:
                with st.spinner("Searching..."):
                    results = st.session_state.model.search(query, **search_params)

                    if results:
                        st.subheader(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            with st.expander(
                                f"Result {i + 1} - "
                                f"Document ID: {result['id']} "
                                f"(Score: {result['score']:.4f})"
                            ):
                                st.text(result["content"])
                    else:
                        st.info("No results found")
        else:
            st.info("Please load data and initialize a model using the sidebar.")

    with tab_stats:
        if st.session_state.model is not None:
            model = st.session_state.model

            st.subheader("General statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model type", st.session_state.model_type)
            with col2:
                st.metric("Document count", len(st.session_state.documents))
            with col3:
                if st.session_state.model_type != "Combined":
                    st.metric("vocab size", len(model.vocabulary))
                else:
                    st.metric("LSA vocab size", len(model.lsa_model.vocabulary))
                    st.metric("Boolean vocab size", len(model.boolean_model.vocabulary))

            st.divider()

            st.subheader("Evaluation metrics")
            if st.button("Evaluate"):
                metrics = compute_metrics(model, search_params)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Precision", round(metrics["P"], 4))
                    st.metric("MAP", round(metrics["MAP"], 4))
                with col2:
                    st.metric("Mean Recall", round(metrics["R"], 4))
                with col3:
                    st.metric("Mean F1-score", round(metrics["F1"], 4))

            st.divider()

            if st.session_state.model_type == "LSA":
                plot_top_terms(model)
            elif st.session_state.model_type == "Boolean":
                plot_term_frequency(model)
            else:
                tab_lsa, tab_boolean = st.tabs(["LSA Component", "Boolean Component"])
                with tab_lsa:
                    plot_top_terms(model.lsa_model)
                with tab_boolean:
                    plot_term_frequency(model.boolean_model)
        else:
            st.info("Please load data and initialize a model using the sidebar.")

    with tab_docs:
        if st.session_state.documents:
            st.header("Document Collection")
            st.text(f"Total documents: {len(st.session_state.documents)}")

            # Show sample documents
            sample_size = min(10, len(st.session_state.documents))
            st.subheader(f"Sample Documents (showing {sample_size})")

            for i in range(sample_size):
                doc_id, content = st.session_state.documents[i]
                with st.expander(f"Document {doc_id}"):
                    st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
        else:
            st.info("No documents loaded. Please load data using the sidebar.")


if __name__ == "__main__":
    main()
