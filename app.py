import streamlit as st

from ui.sidebar import load_sidebar
from ui.utils import load_documents, load_model

st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔍",
    layout="wide",
)


def main() -> None:
    st.title("🔍 Text Retrieval System")

    data_params, model_params, load_button = load_sidebar()
    source_type = data_params.pop("source_type")
    model_type = model_params.pop("model_type")

    # Initialize session state variables
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "model" not in st.session_state:
        st.session_state.model = None

    # Main content area
    tab_search, tab_docs = st.tabs(["Search", "Documents"])

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

            # Search interface based on model type
            query = st.text_input("Enter search query")
            params = {}
            if st.session_state.model_type == "LSA":
                params = {"top_k": st.slider("Number of results", 1, 50, 10)}
            elif st.session_state.model_type == "Combined":
                params = {
                    "boolean_query": st.text_input("Enter Boolean query (optional)"),
                    "mode": st.radio("Search mode", ["boolean_first", "lsa_first"]),
                    "top_k": st.slider("Number of results", 1, 50, 10),
                    "lsa_threshold": st.slider("LSA threshold", 0.0, 1.0, 0.05),
                }

            if st.button("Search") and query:
                with st.spinner("Searching..."):
                    results = st.session_state.model.search(query, **params)

                    if results:
                        st.subheader(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            with st.expander(
                                f"Result {i + 1}: \nDocument ID: {result['id']} \n(Score: {result['score']:.4f})"
                            ):
                                st.text(result["content"])
                    else:
                        st.info("No results found")
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
