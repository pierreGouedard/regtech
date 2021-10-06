# Global import
from typing import Tuple, Any, Dict
import numpy as np
from tensorflow.keras import Model

# Local import
from regtech.datalab.dataops.models.mapping.commit2test import Commit2Test


def train_model(
        ax_features: np.ndarray, ax_targets: np.ndarray, n_kernels: int, kernel_size: int, learning_rate: float,
        nb_epoch: int
) -> Tuple[Model, Dict[str, Any]]:

    # Network parameters
    input_dim, latent_test_dim = ax_features.shape[1:], ax_targets.shape[-1]

    # Instantiate model
    c2t_model = Commit2Test(latent_test_dim, n_kernels, kernel_size, input_dim, learning_rate, nb_epoch)

    # Fit model
    c2t_model.fit(
        np.vstack([ax_features, ax_features, ax_features, ax_features]),
        np.vstack([ax_targets.astype(int), ax_targets.astype(int), ax_targets.astype(int), ax_targets.astype(int)])
    )

    return c2t_model.network, c2t_model.to_dict()
