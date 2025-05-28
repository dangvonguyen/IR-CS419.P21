import streamlit as st

from src.utils import load_csv_data, load_from_directory
from src.models.boolean_model import BooleanModel
from src.models.combined_model import CombinedModel
from src.models.lsa_model import LSAModel

st.set_page_config(
    page_title="Information Retrieval System", page_icon="🔍", layout="wide"
)


def load_model(
    model_type, n_components=None, min_df=None, max_df=None, preprocessing_config=None
):
    factories = {
        "Boolean": lambda: BooleanModel(preprocessing_config=preprocessing_config),
        "LSA": lambda: LSAModel(
            n_components=n_components,
            max_df=max_df,
            min_df=min_df,
            preprocessing_config=preprocessing_config,
        ),
    }
    if model_type != "Combined":
        return factories[model_type]()
    else:
        return CombinedModel(factories["LSA"](), factories["Boolean"]())


def main():
    st.title("🔍 Text Retrieval System")

    # Sidebar for data loading and model configuration
    with st.sidebar:
        st.header("Data Source")
        source_type = st.selectbox("Select data source", ["dir", "csv"])

        if source_type == "dir":
            file_path = st.text_input("Directory path", value="data/train")
            file_pattern = st.text_input("File pattern", value="*.txt")
        else:  # CSV
            file_path = st.text_input("CSV file path")
            text_column = st.text_input("Text column name")
            id_column = st.text_input("ID column name (optional)")

        st.header("Model Configuration")
        model_type = st.selectbox("Select model", ["LSA", "Boolean", "Combined"])

        # LSA model parameters
        n_components, min_df, max_df = None, None, None
        if model_type in ["LSA", "Combined"]:
            st.subheader("LSA Parameters")
            n_components = st.slider("Number of components", 10, 500, 100)
            min_df = st.slider("Minimum document frequency", 1, 10, 1)
            max_df = st.slider("Maximum document frequency (%)", 50, 100, 90) / 100

        # Preprocessing parameters
        st.subheader("Preprocessing")
        preprocessing_config = {
            "lowercase": st.checkbox("Use lowercase", value=True),
            "remove_urls": st.checkbox("Remove URLs", value=True),
            "remove_html": st.checkbox("Remove HTML tags", value=True),
            "remove_special_chars": st.checkbox("Remove special chars", value=True),
            "remove_numbers": st.checkbox("Remove numbers", value=True),
            "remove_stopwords": st.checkbox("Remove stopwords", value=True),
            "stemming": st.checkbox("Use stemming", value=model_type != "Boolean"),
            "lemmatization": st.checkbox("Use lemmatization", value=False),
            "min_word_length": st.slider(
                "Minimum word length", min_value=1, max_value=20, step=1, value=2
            ),
            "custom_stopwords": st.text_input(
                "Custom stopwords (seperated by space)"
            ).split(),
        }

        load_button = st.button("Load Data and Initialize Model")

    # Main content area
    tab1, tab2 = st.tabs(["Search", "Documents"])

    # Initialize session state variables
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None

    # Load data and initialize model when button is pressed
    if load_button:
        with st.spinner("Loading documents and initializing model..."):
            try:
                # Load documents
                if source_type == "dir":
                    documents = load_from_directory(file_path, file_pattern)
                else:
                    documents = load_csv_data(file_path, text_column, id_column)

                if not documents:
                    st.error("No documents loaded. Please check and try again.")
                else:
                    st.session_state.documents = documents
                    st.session_state.model_type = model_type

                    # Initialize the appropriate model
                    model = load_model(
                        model_type, n_components, min_df, max_df, preprocessing_config
                    )
                    model.fit(documents)

                    st.session_state.model = model
                    st.success(
                        f"Successfully loaded {len(documents)} documents and initialized {model_type} model!"
                    )
            except Exception as e:
                st.error(f"Error loading data: {e}")

    # Search tab
    with tab1:
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

    # Documents tab
    with tab2:
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
