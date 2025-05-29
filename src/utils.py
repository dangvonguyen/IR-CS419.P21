import glob
import os
import re
from typing import Literal

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


def load_from_directory(
    directory_path: str,
    return_mode: Literal["doc", "sep"] = "doc",
) -> list[tuple[str, str]] | tuple[list[str], list[str]]:
    """
    Load text files from a directory.

    Args:
        directory_path: Path to the directory containing text files
        return_mode: Mode to return the data in

    Returns:
        If return_mode is "doc", a list of (id, content) tuples.
        If return_mode is "sep", a tuple of two lists - one of ids and one of contents.
    """
    documents = []
    file_paths = glob.glob(os.path.join(directory_path, "*"))
    file_paths.sort(
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", x)
        ]
    )

    for file_path in file_paths:
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            id = os.path.splitext(os.path.basename(file_path))[0]
            documents.append((id, content))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if return_mode == "doc":
        return documents
    else:
        ids, contents = zip(*documents, strict=False)
        return list(ids), list(contents)


def load_csv_data(
    file_path: str,
    return_mode: Literal["doc", "sep"] = "doc",
    sep: str | None = None,
    columns: list[str] | None = None,
    header: Literal["infer"] | None = None,
) -> list[tuple] | tuple[list, ...]:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file
        return_mode: Mode to return the data in
        sep: Separator for the CSV file
        columns: List of column names to load
        header: Infer the column names or not
    """
    df = pd.read_csv(file_path, sep=sep or "\t", header=header, engine="python")

    df.columns = df.columns.astype(str)
    if columns is not None:
        df = df[columns]

    # Convert all columns to strings
    df = df.map(str)

    if return_mode == "doc":
        return [tuple(df.T[col]) for col in df.T.columns]
    else:
        return tuple(df[col].tolist() for col in df.columns)


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_special_chars: bool = True,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    stemming: bool = False,
    lemmatization: bool = False,
    min_word_length: int = 2,
    custom_stopwords: list[str] | None = None,
    return_tokens: bool = False,
) -> str | list[str]:
    if not isinstance(text, str):
        return "" if not return_tokens else []

    # Initialize tools
    stemmer = PorterStemmer() if stemming else None
    lemmatizer = WordNetLemmatizer() if lemmatization else None
    stop_words = set()

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            stop_words.update(custom_stopwords)

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<.*?>", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Process tokens
    processed_tokens = []
    for token in tokens:
        # Remove special characters
        if remove_special_chars:
            token = re.sub(r"[^\w\s]", "", token)

        # Remove numbers
        if remove_numbers:
            token = re.sub(r"\d+", "", token)

        # Skip empty tokens and tokens shorter than min_word_length
        if not token or len(token) < min_word_length:
            continue

        # Remove stopwords
        if remove_stopwords and token in stop_words:
            continue

        # Apply stemming
        if stemmer:
            token = stemmer.stem(token)

        # Apply lemmatization
        if lemmatizer:
            token = lemmatizer.lemmatize(token)

        processed_tokens.append(token)

    if return_tokens:
        return processed_tokens
    else:
        return " ".join(processed_tokens)
