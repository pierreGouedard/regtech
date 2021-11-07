# Global import
from typing  import Tuple
import pandas as pd
import numpy as np

# local import


def build_datasets(path_test: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build fake test datasets.

    Parameters
    ----------
    path_test: str
        Path to fake test like content.

    Returns
    -------

    """
    # Load test data
    df_tests = pd.read_csv(path_test, index_col=None)\
        .sample(frac=1)\
        .reset_index(drop=True)

    # Split tests into historic / master
    df_hist_tests = df_tests.iloc[:10000, :].assign(test_ind=np.arange(10000))
    df_master_tests = df_tests.iloc[10000:10300, :].assign(master_ind=np.arange(300))
    df_val_tests = df_hist_tests.sample(20).assign(
        gid=np.random.randint(0, 3, 20),
        test_ind=np.arange(20)
    )

    return df_hist_tests, df_master_tests, df_val_tests
