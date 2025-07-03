# Information Retrieval System

A text retrieval system implementing various information retrieval models with a Streamlit-based user interface.

## Features

- Multiple retrieval models:
  - Boolean Retrieval Model
  - Vector Space Model (VSM)
  - Latent Semantic Analysis (LSA)
  - Combined Model (Boolean + Vector)
- Interactive search interface
- Document statistics and visualizations
- Model evaluation metrics

## Installation

1. Clone this repository:

```bash
git clone https://github.com/dangvonguyen/IR-CS419.P21.git
cd IR-CS419.P21
```

2. Set up the environment and install dependencies

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Using uv (faster installation)
uv sync
source .venv/bin/activate
```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

### Using the Application

1. **Load Data**: Use the sidebar to select data source and parameters
2. **Select Model**: Choose between Boolean, VSM, LSA, or Combined retrieval models
3. **Search**: Enter queries in the search tab to retrieve relevant documents
4. **Analyze**: View document statistics and model performance in the Statistics tab
5. **Browse**: View loaded documents in the Documents tab

## Project Structure

- `app.py`: Main Streamlit application
- `src/models/`: Implementation of retrieval models
  - `boolean_model.py`: Boolean retrieval with inverted index
  - `vsm_model.py`: Vector Space Model with TF-IDF
  - `lsa_model.py`: Latent Semantic Analysis model
  - `combined_model.py`: Combined Boolean and Vector model
- `src/utils.py`: Utility functions for text processing
- `src/evaluate.py`: Evaluation metrics for retrieval models
- `ui/`: User interface components
