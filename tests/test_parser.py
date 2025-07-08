"""This module contains tests for the parser module."""
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.configuration import PDFParserConfiguration
from components.pdf_parser.pypdf_parser import PyPDFParser


def test_parse_non_existing_pdf():
    config = PDFParserConfiguration({"max_pages": -1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("")
    with pytest.raises(ValueError):
        parser.parse(pdf_path)


def test_parse_non_pdf():
    config = PDFParserConfiguration({"max_pages": -1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/image.png")
    with pytest.raises(ValueError):
        parser.parse(pdf_path)


def test_parse_single_page_pdf():
    config = PDFParserConfiguration({"max_pages": 1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/one_page.pdf")

    result = parser.parse(pdf_path)

    assert result is not None
    assert isinstance(result, str)
    assert result == "Text for testing"


def test_parse_zero_pages():
    config = PDFParserConfiguration({"max_pages": 0})
    pdf_path = Path("tests/data/one_page.pdf")
    with pytest.raises(ValueError):
        parser = PyPDFParser(parser_config=config)
        parser.parse(pdf_path)


def test_parse_empty_pdf():
    config = PDFParserConfiguration({"max_pages": -1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/empty.pdf")

    result = parser.parse(pdf_path)

    assert result is not None
    assert isinstance(result, str)
    assert result == ""


def test_parse_multi_page_pdf_full():
    config = PDFParserConfiguration({"max_pages": -1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/72052.pdf")

    result = parser.parse(pdf_path)
    print(result)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_multi_page_pdf_first_page():
    config = PDFParserConfiguration({"max_pages": 1})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/72052.pdf")

    result = parser.parse(pdf_path)
    print(result)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_parse_too_many_pages():
    config = PDFParserConfiguration({"max_pages": 10000})
    parser = PyPDFParser(parser_config=config)
    pdf_path = Path("tests/data/72052.pdf")

    result = parser.parse(pdf_path)
    print(result)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0