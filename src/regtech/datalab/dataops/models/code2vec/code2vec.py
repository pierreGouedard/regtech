# Global imports
import os.path
from typing import List, Optional
import numpy as np

# Local imports
from .config import PredConfig
from .tensorflow_model import Code2VecModel
from .extractor import Extractor
from regtech.datalab.devops.driver import FileDriver


def predict_vector():
    pass


class Code2VecWrapper(object):
    """
    Code2Vec wrapper that implement predict vector method.

    """
    jar_path = os.path.join(os.path.dirname(__file__), 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar')
    max_path_length = 8
    max_path_width = 2
    n_embeddings = 384

    def __init__(self, model_path: str, on_extraction_error: str = 'skip'):
        self.config = PredConfig(set_defaults=True, load_from_kwargs=True, verify=True, **{"load_path": model_path})
        self.driver = FileDriver('tmp_file', 'Code 2 vec file driver')
        self.model = Code2VecModel(self.config)
        self.path_extractor = Extractor(
            self.config, jar_path=self.jar_path, max_path_length=self.max_path_length,
            max_path_width=self.max_path_width, on_error=on_extraction_error
        )

    def predict_code_vector(self, l_codes: List[str], agg: str = 'mean') -> Optional[np.array]:
        """
        Compute vector from a list of code snippet.

        Parameters
        ----------
        l_codes: list
            Contains the code snippet (str).
        agg: str
            Choose how to aggregate each code snippet.

        Returns
        -------
        array|None

        """
        l_vectors = []
        for code in l_codes:
            # Create tmp files
            tmp_file = self.driver.TempFile(prefix="tmp_", suffix=".java")
            with open(tmp_file.path, 'w') as handle:
                handle.write(code)
            print('_________')
            for l in code.split('\n'):
                print(l)
            print('_________')

            predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(tmp_file.path)

            if predict_lines is not None:
                raw_prediction_results = self.model.predict(predict_lines)
                tmp_file.remove()
                l_vectors.append(np.mean([raw_pred.code_vector for raw_pred in raw_prediction_results], axis=0))

        if not l_vectors:
            return None
        elif agg == "mean":
            return np.mean(l_vectors, axis=0)
        else:
            raise ValueError('agg {} not imlemented'.format(agg))
