from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import preprocess_text


class VSMModel:
    """Vector Space Model (VSM) for information retrieval using TF-IDF."""

    def __init__(
        self,
        max_df: float | int = 1.0,
        min_df: float | int = 1,
        preprocess_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the VSM model.

        Args:
            max_df: Maximum document frequency for terms
            min_df: Minimum document frequency for terms
            preprocessing_config: Preprocessing parameters
        """
        default_config = {
            "lowercase": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_special_chars": True,
            "remove_numbers": True,
            "remove_stopwords": True,
            "stemming": False,
            "lemmatization": True,
            "min_word_length": 2,
            "custom_stopwords": None,
        }

        self.max_df = max_df
        self.min_df = min_df
        self.preprocess_config = {**default_config, **(preprocess_config or {})}

        self._vectorizer = TfidfVectorizer(
            preprocessor=lambda x: preprocess_text(x, **self.preprocess_config),
            lowercase=False,
            stop_words=None,
            max_df=max_df,
            min_df=min_df,
            tokenizer=None,
        )

        self.documents = []
        self.doc_ids = []
        self.document_term_matrix = None
        self.vocabulary = []

        self.is_fitted = False

    def add_document(self, doc_id: str, content: str) -> None:
        """
        Add a document to the collection
        """
        self.documents.append(content)
        self.doc_ids.append(doc_id)
        self.is_fitted = False

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Fit the VSM model to a collection of documents.
        """
        for doc_id, content in documents:
            self.add_document(doc_id, content)

        if not self.documents:
            raise ValueError("No documents added to the system")

        # Convert to TF-IDF vectors
        self.document_term_matrix = self._vectorizer.fit_transform(self.documents)
        self.vocabulary = list[str](self._vectorizer.get_feature_names_out())

        self.is_fitted = True

    def search(
        self, query: str, top_k: int | None = None, threshold: float = 0.05
    ) -> list[dict]:
        """
        Search for relevant documents.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Transform query into TF-IDF space
        query_tfidf = self._vectorizer.transform([query]).toarray()[0]

        # Compute cosine similarity between query and all documents
        similarities = []
        for i, doc_vector in enumerate(self.document_term_matrix.toarray()):
            if np.all(query_tfidf == 0) or np.all(doc_vector == 0):
                similarity = 0.0
            else:
                similarity = 1 - cosine(query_tfidf, doc_vector)

            if similarity >= threshold:
                similarities.append(
                    {
                        "id": self.doc_ids[i],
                        "score": float(similarity),
                        "content": self.documents[i],
                    }
                )

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]

    def get_top_terms_for_document(
        self, doc_id: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Get the top terms for a specific document based on TF-IDF weights.

        Args:
            doc_id: ID of the document to analyze
            top_k: Number of top terms to return

        Returns:
            List of (term, weight) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        if doc_id not in self.doc_ids:
            raise ValueError(f"Document ID {doc_id} not found")

        doc_index = self.doc_ids.index(doc_id)
        doc_vector = self.document_term_matrix[doc_index].toarray().flatten()

        # Get indices of top terms
        top_indices = np.argsort(doc_vector)[::-1][:top_k]

        # Get the actual terms and their weights
        top_terms = [
            (self.vocabulary[i], float(doc_vector[i]))
            for i in top_indices
            if doc_vector[i] > 0
        ]

        return top_terms
