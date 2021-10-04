# Global import
from typing import Optional
import pandas as pd


def extract_test_parameters(create_random_test: bool = True, n_tests: Optional[int] = None) -> pd.DataFrame:
    """

    Args:
        is_test:

    Returns:

    """
    if create_random_test:
        l_parameters, l_values = 'random_string', []
        df_tests = pd.DataFrame()
    else:
        raise ValueError('extracting test is not implemented')

    return df_tests

