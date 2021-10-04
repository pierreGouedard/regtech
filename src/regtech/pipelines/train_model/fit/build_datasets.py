# Global import
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Local import


def build_dataset(
        df_tests: pd.DataFrame, df_commits: pd.DataFrame, df_jira: pd.DataFrame, d_test_params: Dict[int, np.ndarray],
        d_feature_commits: Dict[int, np.ndarray], n_step_commits: int
) -> Tuple[np.ndarray, np.ndarray]:

    import IPython
    IPython.embed()

    # Gather commits by issue key

    # Pad grouped commits to have the same matrix size

    # Arrange test and features to build X and y arrays.


    return np.array([]), np.array([])



