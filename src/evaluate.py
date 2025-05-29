import numpy as np


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: list[str], k: int | None = None
) -> float:
    """
    Calculate precision at rank k.

    Args:
        retrieved_docs: List of document IDs in the order they were retrieved
        relevant_docs: List of document IDs that are actually relevant
        k: Cut-off rank (if None, uses all retrieved documents)

    Returns:
        Precision score between 0 and 1
    """
    # Only consider top-k retrieved documents
    if k is not None:
        retrieved_docs = retrieved_docs[:k]

    if not retrieved_docs:
        return 0.0

    return len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs)


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: list[str], k: int | None = None
) -> float:
    """
    Calculate recall at rank k.

    Args:
        retrieved_docs: List of document IDs in the order they were retrieved
        relevant_docs: List of document IDs that are actually relevant
        k: Cut-off rank (if None, uses all retrieved documents)

    Returns:
        Recall score between 0 and 1
    """
    # Only consider top-k retrieved documents
    if k is not None:
        retrieved_docs = retrieved_docs[:k]

    if not relevant_docs:
        return 0.0

    return len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)


def f1_score(
    retrieved_docs: list[str], relevant_docs: list[str], k: int | None = None
) -> float:
    """
    Calculate F1 score at rank k.

    Args:
        retrieved_docs: List of document IDs in the order they were retrieved
        relevant_docs: List of document IDs that are actually relevant
        k: Cut-off rank (if None, uses all retrieved documents)

    Returns:
        F1 score between 0 and 1
    """
    precision = precision_at_k(retrieved_docs, relevant_docs, k)
    recall = recall_at_k(retrieved_docs, relevant_docs, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average_precision(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    """
    Calculate average precision.

    Args:
        retrieved_docs: List of document IDs in the order they were retrieved
        relevant_docs: List of document IDs that are actually relevant

    Returns:
        Average precision score between 0 and 1
    """
    if not retrieved_docs or not relevant_docs:

        return 0.0

    relevant_set = set(relevant_docs)
    precision_sum = 0.0
    relevant_found = 0

    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / len(relevant_docs)


def mean_average_precision(
    query_results: dict[str, tuple[list[str], list[str]]],
) -> float:
    """
    Calculate mean average precision.

    Args:
        query_results: Dictionary of query results

    Returns:
        Mean average precision score between 0 and 1
    """
    if not query_results:
        return 0.0

    ap_scores = []
    for retrieved_docs, relevant_docs in query_results.values():
        ap = average_precision(retrieved_docs, relevant_docs)
        ap_scores.append(ap)

    return np.mean(ap_scores)


def comprehensive_evaluation(
    query_results: dict[str, tuple[list[str], list[str]]], k: int | None = None
) -> dict[str, float]:
    """ Calculate comprehensive evaluation metrics.

    Args:
        query_results: Dictionary of query_id -> (retrieved_docs, relevant_docs)
        k: Cut-off rank (if None, uses all retrieved documents)

    Returns:
        Dictionary of evaluation metrics
    """
    results = {
        "MAP": mean_average_precision(query_results),
        "P": np.mean([precision_at_k(r[0], r[1], k) for r in query_results.values()]),
        "R": np.mean([recall_at_k(r[0], r[1], k) for r in query_results.values()]),
        "F1": np.mean([f1_score(r[0], r[1], k) for r in query_results.values()]),
    }
    return results
