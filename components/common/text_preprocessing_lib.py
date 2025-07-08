"""This module contains a class for cleaning a text data."""
import re
import string

import nltk


STOP_WORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess_single_text(text: str) -> str:
    """Performs preprocessing of a single text.

    Lowers the text, removes web links, emails, numbers, punctuation. Then lemmatizes the text and
    removes stop words.

    Args:
        text: Text to preprocess.

    Returns:
        Preprocessed text as a single string.
    """
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove web links
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)  # Remove emails
    text = text.translate(str.maketrans({char: ' ' for char in (string.punctuation + "«»–…")}))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOP_WORDS]
    return ' '.join(tokens)