"""This module contains a class which is used to use classification model in a reference mode."""
from warnings import warn

import joblib
import numpy as np
import torch

from common.configuration import ModelInferenceConfiguration
from components.classification_model.model import FFNN
from components.common.text_preprocessing_lib import preprocess_single_text


target_values_to_names = {
    0: 'alt.atheism',
    1: 'comp.graphics',
    2: 'comp.os.ms-windows.misc',
    3: 'comp.sys.ibm.pc.hardware',
    4: 'comp.sys.mac.hardware',
    5: 'comp.windows.x',
    6: 'misc.forsale',
    7: 'rec.autos',
    8: 'rec.motorcycles',
    9: 'rec.sport.baseball',
    10: 'rec.sport.hockey',
    11: 'sci.crypt',
    12: 'sci.electronics',
    13: 'sci.med',
    14: 'sci.space',
    15: 'soc.religion.christian',
    16: 'talk.politics.guns',
    17: 'talk.politics.mideast',
    18: 'talk.politics.misc',
    19: 'talk.religion.misc'
}


def check_and_choose_torch_device() -> torch.device:
    """Returns available device for hosting torch data."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ModelInferencePipeline:
    def __init__(self, model_inference_config: ModelInferenceConfiguration):
        path_to_model_weights = model_inference_config.path_to_model_weights
        path_to_tf_idf_vectorizer = model_inference_config.path_to_tf_idf_vectorizer
        if not path_to_model_weights.exists():
            raise FileNotFoundError(f"Model weights not found at {path_to_model_weights}")
        if not path_to_tf_idf_vectorizer.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer file"
                                    f" not found at {path_to_tf_idf_vectorizer}")
        self._device = check_and_choose_torch_device()
        model_checkpoint = torch.load(path_to_model_weights)
        model_args = model_checkpoint["model_args"]

        self._model = FFNN(**model_args).to(self._device)

        self._model.load_state_dict(model_checkpoint['model_state_dict'])
        self._model.eval()
        self._tfidf_vectorizer = joblib.load(path_to_tf_idf_vectorizer)

    def _prepare_and_vectorize_text(self, text: str) -> torch.Tensor:
        """Performs preprocessing of the text and vectorizes it using TF-IDF vectorizer.

        Args:
            text: String text to vectorize.

        Returns:
            TF-IDF vector of the text as a torch.Tensor.
        """
        tf_idf_vector = self._tfidf_vectorizer.transform([text]).toarray().astype(np.float32)
        tf_idf_vector = torch.from_numpy(tf_idf_vector).to(self._device)
        if tf_idf_vector.sum().item() == 0:
            warn("No words in the text were found in the vocabulary. "
                 "The text will be treated as a vector of zeros. Prediction may be incorrect.")
        return tf_idf_vector

    def _predict_class_on_vector(self, tf_idf_vector: torch.Tensor) -> str:
        """Performs prediction on the given vector.

        Args:
            tf_idf_vector: TF-IDF vector for performing prediction.

        Returns:
            String class name.
        """
        model_prediction = self._model(tf_idf_vector).argmax().item()
        return target_values_to_names[model_prediction]

    def run(self, text: str) -> str:
        """Performs prediction on the given text.

        Args:
            text: String text to predict.

        Returns:
            Class name for the predicted value as a string.
        """
        preprocessed_text = self._prepare_and_vectorize_text(text)
        class_name = self._predict_class_on_vector(preprocessed_text)
        return class_name
