# Global import
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# local import
from regtech.datalab.dataops.models.dictionary_reg.lasso import LassoDictRegression


def get_master_tests(
        ax_validation_tests: np.ndarray, ax_master_tests: np.ndarray, df_validation_tests: pd.DataFrame,
        df_master_tests: pd.DataFrame, d_dict_reg_params: Dict[str, Any]
):
    # Build mixtures
    l_mixtures = []
    for _, df_val_sub in df_validation_tests.groupby("gid"):
        l_mixtures.append(ax_validation_tests[df_val_sub.test_ind.values, :].sum(axis=0))

    ax_mixtures = np.vstack(l_mixtures)
    ax_mixtures /= ax_mixtures.sum(axis=1, keepdims=True)

    # Instantiate Dictionary regressor
    dreg = LassoDictRegression(d_dict_reg_params, ax_master_tests.T)

    # get description of model for different alpha
    d_info = dreg.describe_alpha(ax_mixtures)
    print(d_info)

    # Fit model and score master tests
    l_df_scored_masters = []
    for i, ax_mixture in enumerate(ax_mixtures):
        l_df_scored_masters.append(
            dreg.infer_coefs(ax_mixture).score_atoms(df_master_tests.copy(), 'master_ind', True)
                .assign(mixture_ind=i)
        )

    return pd.concat(l_df_scored_masters)


