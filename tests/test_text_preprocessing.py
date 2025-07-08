"""This module contains tests for the text preprocessing function."""
import string
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.common.text_preprocessing_lib import preprocess_single_text


def test_text_preprocessing_lines_splits():
    input_data = "\nuseful\ntext\n\n\n\n"
    output_data = "useful text"

    preprocessed_text = preprocess_single_text(input_data)

    assert preprocessed_text is not None
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == output_data


def test_text_preprocessing_empty():
    input_data = ""
    output_data = ""

    preprocessed_text = preprocess_single_text(input_data)

    assert preprocessed_text is not None
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == output_data


def test_text_preprocessing_single_space():
    input_data = " "
    output_data = ""

    preprocessed_text = preprocess_single_text(input_data)

    assert preprocessed_text is not None
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == output_data


def test_text_preprocessing_punctuation():
    input_data = "useful, text!? << >>" + string.punctuation
    output_data = "useful text"

    preprocessed_text = preprocess_single_text(input_data)

    assert preprocessed_text is not None
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == output_data


def test_text_preprocessing_numbers_emails_links():
    input_data = ("useful text sent from the user 123123123 from the link "
                  "https://www.google.com, https://www.google.com or www.google.com,"
                  " the email test@test.com")
    output_data = "useful text sent user link email"

    preprocessed_text = preprocess_single_text(input_data)

    assert preprocessed_text is not None
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == output_data
