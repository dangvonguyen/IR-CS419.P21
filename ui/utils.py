from typing import Any, Literal

from src.models import BooleanModel, CombinedModel, LSAModel
from src.utils import load_csv_data, load_from_directory


def load_model(
    model_type: Literal["LSA", "Boolean", "Combined"],
    n_components: int | None = None,
    min_df: int | None = None,
    max_df: float | None = None,
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
        ),
    }
    if model_type != "Combined":
        return factories[model_type]()
    else:
        return CombinedModel(factories["LSA"](), factories["Boolean"]())


def load_documents(
    source_type: Literal["dir", "csv"],
    file_path: str,
    file_pattern: str | None = None,
    text_column: str | None = None,
    id_column: str | None = None,
) -> list[tuple[str, str]]:
    if source_type == "dir":
        return load_from_directory(file_path, file_pattern)
    else:
        return load_csv_data(file_path, text_column, id_column)
