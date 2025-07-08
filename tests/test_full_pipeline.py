"""This module contains tests for testing all the pipeline stages as one."""
from pathlib import Path
import sys
import subprocess

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

def test_main_end_to_end():
    script_path = Path("main.py")
    config_path = Path("tests/resources/test_config.yaml")
    pdf_path = Path("tests/data/72052.pdf")

    result = subprocess.run(
        ["python", str(script_path), "-c", str(config_path), "-p", str(pdf_path)],
        capture_output=True,
        text=True,
        check=True
    )

    assert "Class name for the provided file is" in result.stdout
