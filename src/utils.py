import glob
import os
import re
from typing import Any, Optional

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
    directory_path: str, file_pattern: str = "*.txt"
) -> list[tuple[str, str]]:
    """
    Load text files from a directory.

    Args:
        directory_path: Path to the directory containing text files
        file_pattern: Pattern for file selection

    Returns:
        List of tuples (id, content)
    """
    documents = []
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    file_paths.sort(
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", x)
        ]
    )

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            id = os.path.splitext(os.path.basename(file_path))[0]
            documents.append((id, content))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents


def load_csv_data(
    file_path: str, text_column: str, id_column: Optional[str] = None
) -> list[tuple[Any, str]]:
    """
    Load document data from a CSV file.

    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing document text
        id_column: Name of the column containing document IDs (or None to use index)

    Returns:
        List of tuples (id, content)
    """
    df = pd.read_csv(file_path)

    if id_column is None:
        return list(zip(df.index.tolist(), df[text_column].tolist()))
    else:
        return list(zip(df[id_column].tolist(), df[text_column].tolist()))


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
