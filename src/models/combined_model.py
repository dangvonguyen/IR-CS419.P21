from typing import Literal

from src.models.boolean_model import BooleanModel
from src.models.lsa_model import LSAModel


class CombinedModel:
    """Combined model that integrates LSA and Boolean model."""

    def __init__(
        self,
        lsa_model: LSAModel | None = None,
        boolean_model: BooleanModel | None = None,
    ) -> None:
        """
        Initialize the combined model.

        Args:
            lsa_model: Initialized LSA model (or None to create a new one)
            boolean_model: Initialized Boolean model (or None to create a new one)
        """
        self.lsa_model = lsa_model if lsa_model else LSAModel()
        self.boolean_model = boolean_model if boolean_model else BooleanModel()

        self.documents = None

        self.is_fitted = False

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Fit both models to the document collection.
        """
        self.documents = documents
        self.lsa_model.fit(documents)
        self.boolean_model.fit(documents)

        self.is_fitted = True

    def search(
        self,
        query: str,
        boolean_query: str | None = None,
        mode: Literal["lsa_first", "boolean_first"] = "boolean_first",
        top_k: int | None = None,
        lsa_threshold: float = 0.05,
    ) -> list[dict]:
        """
        Search for documents matching the query and optional Boolean query.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if mode == "lsa_first":
            return self._lsa_first(query, boolean_query, top_k, lsa_threshold)
        elif mode == "boolean_first":
            return self._boolean_first(query, boolean_query, top_k, lsa_threshold)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _lsa_first(
        self, query: str, boolean_query: str | None, top_k: int | None, threshold: float
    ) -> list[dict]:
        """
        Apply LSA search first, get top resutls, then filter with Boolean query.
        """
        # Get LSA results
        lsa_results = self.lsa_model.search(query, top_k=top_k)
        lsa_results = [r for r in lsa_results if r["score"] >= threshold]

        if not boolean_query or not boolean_query.strip():
            for result in lsa_results:
                result["mode"] = "lsa_first"
                result["boolean_match"] = False
            return lsa_results

        # Get Boolean results for filtering
        boolean_results = self.boolean_model.search(boolean_query)
        boolean_ids = {r["id"] for r in boolean_results}

        filtered_results = []
        for result in lsa_results:
            if result["id"] in boolean_ids:
                result["mode"] = "lsa_first"
                result["boolean_match"] = True
                filtered_results.append(result)

        return filtered_results

    def _boolean_first(
        self, query: str, boolean_query: str | None, top_k: int | None, threshold: float
    ) -> list[dict]:
        """
        Apply Boolean search first, then rank with LSA.
        """
        # Get Boolean results
        boolean_results = self.boolean_model.search(boolean_query)

        if not boolean_results:
            return []

        # Get LSA scores for all documents
        lsa_results = self.lsa_model.search(query, top_k=None)
        lsa_scores = {r["id"]: r["score"] for r in lsa_results}

        # Enhanc results with LSA scores
        enhanced_results = []
        for result in boolean_results:
            doc_id = result["id"]
            lsa_score = lsa_scores.get(doc_id, 0.0)
            if lsa_score < threshold:
                continue

            enhanced_results.append(
                {
                    "id": doc_id,
                    "score": lsa_score,
                    "content": result["content"],
                    "mode": "boolean_first",
                    "boolean_match": True,
                }
            )

        # Sort by LSA score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        return enhanced_results[:top_k]
