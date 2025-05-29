from typing import Any, Literal

from src.evaluate import comprehensive_evaluation
from src.models import BooleanModel, CombinedModel, LSAModel
from src.utils import load_csv_data, load_from_directory


def load_model(
    model_type: Literal["LSA", "Boolean", "Combined"],
    n_components: int | None = None,
    min_df: int | None = None,
    max_df: float | None = None,
    random_state: int | None = None,
    preprocess_config: dict[str, Any] | None = None,
    boolean_config: dict[str, Any] | None = None,
) -> BooleanModel | LSAModel | CombinedModel:
    factories = {
        "Boolean": lambda: BooleanModel(
            preprocess_config=boolean_config or preprocess_config
        ),
        "LSA": lambda: LSAModel(
            n_components=n_components,
            max_df=max_df,
            min_df=min_df,
            preprocess_config=preprocess_config,
            random_state=random_state,
        ),
    }
    if model_type != "Combined":
        return factories[model_type]()
    else:
        return CombinedModel(factories["LSA"](), factories["Boolean"]())


def load_documents(
    source_type: Literal["dir", "csv"],
    path: str,
    sep: str | None = None,
    columns: list[str] | None = None,
    header: Literal["infer"] | None = None,
) -> list[tuple[str, str]]:
    if source_type == "dir":
        return load_from_directory(path)
    else:
        return load_csv_data(path, sep=sep, columns=columns, header=header)


def compute_metrics(
    model: BooleanModel | LSAModel | CombinedModel,
    search_params: dict[str, Any],
    query_path: str = "data/TEST/query.txt",
    res_path: str = "data/TEST/RES",
) -> dict[str, float]:
    queries = load_csv_data(query_path)

    query_results = {}
    for query_id, query in queries:
        retrieved_docs = [doc["id"] for doc in model.search(query, **search_params)]
        relevant_docs, = load_csv_data(
            f"{res_path}/{query_id}.txt",
            return_mode="sep",
            sep=r"[ \t]+",
            columns=["1"],
        )
        query_results[query_id] = (retrieved_docs, relevant_docs)

    return comprehensive_evaluation(query_results)
