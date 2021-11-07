# Global import
from typing import Tuple
import pandas as pd
import numpy as np

# local import
from regtech.datalab.dataops.text.utils import load_stopwords
from regtech.datalab.dataops.text.fuzzy_bow import FuzzyBagOfWords


def encoder_fit_transform(
        df_historical_tests: pd.DataFrame, df_master_tests: pd.DataFrame, df_validation_tests: pd.DataFrame,
        path_stopwords: str, threshold_inter: float, threshold_levenstein: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FuzzyBagOfWords]:
    """

    Parameters
    ----------
    df_historical_tests
    df_master_tests
    df_validation_tests
    path_stopwords
    threshold_inter
    threshold_levenstein

    Returns
    -------

    """
    # Load stopwords
    d_stopwords = load_stopwords(path_stopwords)

    # Instantiate and fit fuzzy bag of words
    fuzzy_bow = FuzzyBagOfWords(threshold_inter, threshold_levenstein, d_stopwords)\
        .fit(df_historical_tests, ['text'])

    # Transform tests
    ax_hist_test_encoded = fuzzy_bow.transform(df_historical_tests, ['text'])
    ax_master_test_encoded = fuzzy_bow.transform(df_master_tests, ['text'])
    ax_val_test_encoded = fuzzy_bow.transform(df_validation_tests, ['text'])

    return ax_hist_test_encoded, ax_master_test_encoded, ax_val_test_encoded, fuzzy_bow
