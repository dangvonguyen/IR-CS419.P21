from typing import Literal

from src.models.boolean_model import BooleanModel
from src.models.lsa_model import LSAModel
from src.models.vsm_model import VSMModel


class CombinedModel:
    """Combined model that integrates VSM/LSA and Boolean model."""

    def __init__(
        self,
        vector_model_type: Literal["LSA", "VSM"] = "LSA",
        vector_model: LSAModel | VSMModel | None = None,
        boolean_model: BooleanModel | None = None,
    ) -> None:
        """
        Initialize the combined model.

        Args:
            vector_model_type: Which vector model to use ("LSA" or "VSM")
            vector_model: Initialized vector model (or None to create a new one)
            boolean_model: Initialized Boolean model (or None to create a new one)
        """
        self.vector_model_type = vector_model_type

        if vector_model:
            self.vector_model = vector_model
        else:
            if vector_model_type == "LSA":
                self.vector_model = LSAModel()
            else:  # VSM
                self.vector_model = VSMModel()

        self.boolean_model = boolean_model if boolean_model else BooleanModel()

        self.documents = None

        self.is_fitted = False

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Fit both models to the document collection.
        """
        self.documents = documents
        self.vector_model.fit(documents)
        self.boolean_model.fit(documents)

        self.is_fitted = True

    def search(
        self,
        query: str,
        boolean_query: str | None = None,
        search_mode: Literal["vector_first", "boolean_first"] = "boolean_first",
        top_k: int | None = None,
        threshold: float = 0.05,
    ) -> list[dict]:
        """
        Search for documents matching the query and optional Boolean query.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if search_mode == "vector_first":
            return self._vector_first(query, boolean_query, top_k, threshold)
        elif search_mode == "boolean_first":
            return self._boolean_first(query, boolean_query, top_k, threshold)
        else:
            raise ValueError(f"Unknown mode: {search_mode}")

    def _vector_first(
        self, query: str, boolean_query: str | None, top_k: int | None, threshold: float
    ) -> list[dict]:
        """
        Apply vector model search first, get top results, then filter with Boolean query.
        """
        # Get vector model results
        vector_results = self.vector_model.search(query, top_k, threshold)
        vector_results = [r for r in vector_results if r["score"]]

        if not boolean_query or not boolean_query.strip():
            for result in vector_results:
                result["mode"] = "vector_first"
                result["boolean_match"] = False
            return vector_results

        # Get Boolean results for filtering
        boolean_results = self.boolean_model.search(boolean_query)
        boolean_ids = {r["id"] for r in boolean_results}

        filtered_results = []
        for result in vector_results:
            if result["id"] in boolean_ids:
                result["mode"] = "vector_first"
                result["boolean_match"] = True
                filtered_results.append(result)

        return filtered_results

    def _boolean_first(
        self, query: str, boolean_query: str | None, top_k: int | None, threshold: float
    ) -> list[dict]:
        """
        Apply Boolean search first, then rank with vector model.
        """
        # Get Boolean results
        boolean_results = self.boolean_model.search(boolean_query)

        if not boolean_results:
            return []

        # Get vector model scores for all documents
        vector_results = self.vector_model.search(query, top_k=None)
        vector_scores = {r["id"]: r["score"] for r in vector_results}

        # Enhance results with vector model scores
        enhanced_results = []
        for result in boolean_results:
            doc_id = result["id"]
            vector_score = vector_scores.get(doc_id, 0.0)
            if vector_score < threshold:
                continue

            enhanced_results.append(
                {
                    "id": doc_id,
                    "score": vector_score,
                    "content": result["content"],
                    "mode": "boolean_first",
                    "boolean_match": True,
                }
            )

        # Sort by vector model score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        return enhanced_results[:top_k]
