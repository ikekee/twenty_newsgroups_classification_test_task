"""This module contains a main script for inferencing the classification model on the pdf data."""
import argparse
from pathlib import Path

from common.configuration import Configuration
from components.inference_pipeline.inference_pipeline import ModelInferencePipeline
from components.pdf_parser.pypdf_parser import PyPDFParser


def main(config_path: Path, pdf_file_path: Path):
    config = Configuration(config_path)
    pdf_parser = PyPDFParser(config.pdf_parser_configuration)
    inference_pipeline = ModelInferencePipeline(config.model_inference_configuration)

    text = pdf_parser.parse(pdf_file_path)
    if not text:
        raise ValueError("No text found in the PDF file.")
    class_name = inference_pipeline.run(text)
    print(f"Class name for the provided file is '{class_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config_path',
        metavar='path/to/file',
        type=Path,
        help='A path to the yaml configuration file. '
             f'Default is config.yaml in the root folder of the project.',
        default=Path('config.yaml')
    )
    parser.add_argument(
        '-p',
        '--pdf_file_path',
        metavar='path/to/file',
        type=Path,
        help='A path to the pdf file to be classified.',
        required=True
    )
    args = parser.parse_args()
    main(args.config_path, args.pdf_file_path)
