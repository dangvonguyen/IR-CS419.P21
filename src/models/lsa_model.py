from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import preprocess_text


class LSAModel:
    """Latent Semantic Analysis (LSA) model for information retrieval."""

    def __init__(
        self,
        n_components: int = 100,
        max_df: float | int = 1.0,
        min_df: float | int = 1,
        preprocess_config: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize the LSA model.

        Args:
            n_components: Number of latent sematic dimensions (topics) to extract
            max_df: Maximum document frequency for terms
            min_df: Minimum document frequency for terms
            preprocessing_config: Preprocessing parameters
            random_state: Random seed for reproducibility
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

        self.n_components = n_components
        self.max_df = max_df
        self.min_df = min_df
        self.preprocess_config = {**default_config, **(preprocess_config or {})}
        self.random_state = random_state

        self._vectorizer = TfidfVectorizer(
            preprocessor=lambda x: preprocess_text(x, **self.preprocess_config),
            lowercase=False,
            stop_words=None,
            max_df=max_df,
            min_df=min_df,
            tokenizer=None,
        )

        self._svd_model = TruncatedSVD(n_components, random_state=random_state)

        self.documents = []
        self.doc_ids = []
        self.document_term_matrix = None
        self.document_vectors = None
        self.term_vectors = None
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
        Fit the LSA model to a collection of documents.
        """
        for doc_id, content in documents:
            self.add_document(doc_id, content)

        if not self.documents:
            raise ValueError("No documents added to the system")

        # Convert to TF-IDF vectors
        self.document_term_matrix = self._vectorizer.fit_transform(self.documents)
        self.vocabulary = list[str](self._vectorizer.get_feature_names_out())

        # Apply SVD
        self.document_vectors = self._svd_model.fit_transform(self.document_term_matrix)
        self.term_vectors = self._svd_model.components_.T
        self.singlar_values = self._svd_model.singular_values_

        self.is_fitted = True

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search for relevant documents.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        # Transform query into TF-IDF space
        query_tfidf = self._vectorizer.transform([query])

        # Transform query into LSA space
        query_semantic = self._svd_model.transform(query_tfidf).flatten()

        # Compute cosine similarity between query and all documents
        similarities = []
        for i, doc_vector in enumerate(self.document_vectors):
            similarity = 1 - cosine(query_semantic, doc_vector)
            similarities.append(
                {
                    "id": self.doc_ids[i],
                    "score": float(similarity),
                    "content": self.documents[i],
                }
            )

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]

    def search_related_terms(self, term: str, top_k: int | None = None) -> list[dict]:
        """
        Search for related terms.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        if term not in self.vocabulary:
            return []

        term_index = np.where(self.vocabulary == term)[0][0]
        target_vector = self.term_vectors[term_index]

        # Compute cosine similarity with all terms in the corpus
        similarities = []
        for i, term_vector in enumerate(self.term_vectors):
            if i != term_index:
                similarity = 1 - cosine(target_vector, term_vector)
                similarities.append({"term": self.vocabulary[i], "score": similarity})

        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
