"""This module contains tests for the inference pipeline consisting of vectorizer and the model."""
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.configuration import ModelInferenceConfiguration
from components.inference_pipeline.inference_pipeline import ModelInferencePipeline
from components.inference_pipeline.inference_pipeline import target_values_to_names


def test_inference_pipeline_base_case():
    config = ModelInferenceConfiguration({
        "path_to_model_weights": "models/best.pth",
        "path_to_tf_idf_vectorizer": "models/tfidf_vectorizer.pkl"
    })
    model_inference_pipeline = ModelInferencePipeline(model_inference_config=config)
    input_text = "some text which is used for prediction"
    result = model_inference_pipeline.run(input_text)

    assert result is not None
    assert isinstance(result, str)
    assert result in target_values_to_names.values()


def test_inference_pipeline_tokes_not_from_vocab():
    config = ModelInferenceConfiguration({
        "path_to_model_weights": "models/best.pth",
        "path_to_tf_idf_vectorizer": "models/tfidf_vectorizer.pkl"
    })
    model_inference_pipeline = ModelInferencePipeline(model_inference_config=config)
    input_text = "Ωåßœ∑"

    with pytest.warns(UserWarning):
        result = model_inference_pipeline.run(input_text)

    assert result is not None
    assert isinstance(result, str)
    assert result == "unknown"
