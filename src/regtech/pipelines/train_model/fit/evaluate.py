# Global import
from typing import Tuple, Any, Dict
import numpy as np
from tensorflow.keras import Model

# Local import
from regtech.datalab.dataops.models.mapping.commit2test import Commit2Test


def evaluate(commit2test_network: Model, commit2test_params: Dict[str, Any]) -> None:
    """
    Evaluate model.

    Parameters
    ----------
    commit2test_network: Model

    commit2test_params: dict

    Returns
    -------

    """
    # Instantiate model
    c2t_model = Commit2Test.from_dict(commit2test_params, commit2test_network)

    # Evaluate



