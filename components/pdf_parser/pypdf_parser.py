"""This module contains a class for parsing PDF files using pypdf."""
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

from common.configuration import PDFParserConfiguration


class PyPDFParser:
    """This class contains a functionality for parsing PDF files using pypdf.

    Attributes:
        _max_pages: Maximum number of pages to read from the PDF file.
    """

    def __init__(self, parser_config: PDFParserConfiguration):
        """Creates an instance of the class.

        Args:
            parser_config: PDFParserConfiguration object with the parser settings.
        """
        if parser_config.max_pages == 0:
            raise ValueError("Max pages cannot be 0. Set to a positive integer "
                             "or -1 to read all pages.")
        self._max_pages = parser_config.max_pages

    def parse(self, pdf_file_path: Path) -> Optional[str]:
        """Extracts text from a PDF file from a provided path.

        Args:
            pdf_file_path: Path to the PDF file for parsing.

        Returns:
            The parsed text from the PDF file if it can be read, None otherwise.
        """
        if not pdf_file_path.exists() or pdf_file_path.suffix != ".pdf":
            raise ValueError(
                "Invalid file path or file is not a PDF. "
                "Please provide a valid PDF file path."
            )
        reader = PdfReader(pdf_file_path)
        num_pages = self._max_pages if self._max_pages != -1 else len(reader.pages)
        result = []
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            result.append(text)
        if not result:
            return None
        return "\n".join(result)
