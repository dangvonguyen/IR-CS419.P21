import re
from collections import defaultdict
from typing import Any

from src.utils import preprocess_text


class BooleanModel:
    """A Boolean Model implementation for information retrieval."""

    def __init__(self, preprocess_config: dict[str, Any] | None = None) -> None:
        default_config = {
            "lowercase": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_special_chars": True,
            "remove_numbers": True,
            "remove_stopwords": True,
            "stemming": False,
            "lemmatization": False,
            "min_word_length": 2,
            "custom_stopwords": None,
        }

        self.preprocessing_config = {**default_config, **(preprocess_config or {})}
        self.preprocessing_config["return_tokens"] = True

        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.vocabulary: set[str] = set()
        self.inverted_index: dict[str, set[str]] = defaultdict(set)

        self._id_to_index: dict[str, int] = {}
        self._all_doc_ids: set[str] = set()

    def add_document(self, doc_id: str, content: str) -> None:
        """
        Add a single document to the index.
        """
        if doc_id in self._id_to_index:
            raise ValueError(f"Document with ID '{doc_id}' already exists")

        self.documents.append(content)
        self.doc_ids.append(doc_id)
        self._id_to_index[doc_id] = len(self.doc_ids) - 1
        self._all_doc_ids.add(doc_id)

        # Preprocess content
        terms = preprocess_text(content, **self.preprocessing_config)

        # Update inverted index
        for term in set(terms):
            self.vocabulary.add(term)
            self.inverted_index[term].add(doc_id)

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Build the index from a list of (doc_id, content) tuples.
        """
        for doc_id, content in documents:
            self.add_document(doc_id, content)

    def _parse_query(self, query: str) -> set[str]:
        if not query.strip():
            return set()

        tokens = re.split(r"\b(AND|OR|NOT)\b", query)
        tokens = [
            t.lower().strip() if t not in ["AND", "OR", "NOT"] else t
            for t in tokens
            if t.strip()
        ]

        if not tokens:
            return set()

        # Start with the first term
        if tokens[0] == "NOT":
            if len(tokens) > 1:
                result = self._all_doc_ids - self.inverted_index.get(tokens[1], set())
                start_idx = 2
            else:
                return set()
        else:
            result = self.inverted_index.get(tokens[0], set())
            start_idx = 1

        # Process remaining tokens in pairs (operator, term)
        i = start_idx
        while i < len(tokens) - 1:
            operator, term = tokens[i], tokens[i + 1]

            term_docs = self.inverted_index.get(term, set())

            if operator == "AND":
                result = result.intersection(term_docs)
            elif operator == "OR":
                result = result.union(term_docs)
            elif operator == "NOT":
                result = result - term_docs

            i += 2

        return result

    def search(self, query: str | None) -> list[dict]:
        """
        Search for documents matching a Boolean query.
        """
        # Get document IDs that match the query
        if query and query.strip():
            matching_doc_ids = self._parse_query(query)
        else:
            matching_doc_ids = self.doc_ids

        # Retrieve relevant documents
        results = []
        for doc_id in matching_doc_ids:
            if doc_id in self._id_to_index:
                content = self.documents[self._id_to_index[doc_id]]
                results.append({"id": doc_id, "content": content, "score": 1.0})

        return results
