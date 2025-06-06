from typing import Any

import streamlit as st


def load_sidebar() -> tuple[dict[str, Any], dict[str, Any], bool]:
    # Sidebar for data loading and model configuration
    with st.sidebar:
        st.header("Data Source")
        source_type = st.selectbox("Select data source", ["dir", "csv"])

        # Data parameters
        data_params = {"source_type": source_type}
        if source_type == "dir":
            data_params["path"] = st.text_input("Directory path", value="data/Cranfield")  # fmt: skip
        else:
            data_params["path"] = st.text_input("CSV file path")
            data_params["sep"] = st.text_input("Separator", value=r"\t")
            data_params["columns"] = st.text_input("Columns")
            data_params["header"] = st.selectbox("Header", ["infer", None])

        st.header("Model Configuration")
        model_type = st.selectbox("Select model", ["LSA", "VSM", "Boolean", "Combined"])

        # Model parameters
        model_params = {"model_type": model_type}
        if model_type == "Combined":
            model_params["vector_model_type"] = st.selectbox("Select vector model", ["LSA", "VSM"])  # fmt: skip

        if model_type != "Boolean":
            st.subheader("Parameters")
            model_params["min_df"] = st.slider("Minimum document frequency", 1, 10, 1)
            model_params["max_df"] = st.slider("Maximum document frequency (%)", 50, 100, 90) / 100  # fmt: skip

            if model_type == "LSA" or model_params.get("vector_model_type") == "LSA":
                model_params["n_components"] = st.slider("Number of components", 10, 500, 100)  # fmt: skip
                col1, col2 = st.columns(2, vertical_alignment="bottom")
                with col1:
                    use_random = st.checkbox("Random state", value=True)
                with col2:
                    random_state = st.number_input("Random state", value=0, disabled=not use_random)  # fmt: skip
                    if use_random:
                        model_params["random_state"] = random_state

        # Preprocessing parameters
        st.subheader("Preprocessing")
        if model_type != "Combined":
            model_params["preprocess_config"] = load_preprocess_config(model_type)
        else:
            tab_vector, tab_boolean = st.tabs(["Vector Config", "Boolean Config"])
            with tab_vector:
                model_params["preprocess_config"] = load_preprocess_config("Vector")
            with tab_boolean:
                model_params["boolean_config"] = load_preprocess_config("Boolean", " ")

        load_button = st.button("Load Data and Model")

    return data_params, model_params, load_button


def load_preprocess_config(model_type: str, extra: str = "") -> dict[str, Any]:
    return {
        "lowercase": st.checkbox(f"Use lowercase{extra}", value=True),
        "remove_urls": st.checkbox(f"Remove URLs{extra}", value=True),
        "remove_html": st.checkbox(f"Remove HTML tags{extra}", value=True),
        "remove_special_chars": st.checkbox(f"Remove special chars{extra}", value=True),
        "remove_numbers": st.checkbox(f"Remove numbers{extra}", value=True),
        "remove_stopwords": st.checkbox(f"Remove stopwords{extra}", value=True),
        "stemming": st.checkbox(f"Use stemming{extra}", value=model_type != "Boolean"),
        "lemmatization": st.checkbox(f"Use lemmatization{extra}", value=False),
        "min_word_length": st.slider(f"Minimum word length{extra}", 1, 20, 1, 1),
        "custom_stopwords": st.text_input(f"Custom stopwords (separated by space){extra}").split(),
    }  # fmt: skip
