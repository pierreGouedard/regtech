# local import
from typing import Optional
import pandas as pd
from random import randint


def extract_jira_info(
        create_random_join_info: bool, n_tests: Optional[int] = None, n_commit_group: Optional[int] = None
) -> pd.DataFrame:
    """

    Args:
        create_random_join_info:
        n_tests:
        n_commit_group:

    Returns:

    """
    if create_random_join_info:
        df_jira = pd.DataFrame([
            {"commit_group": randint(0, n_commit_group - 1), "test_id": randint(0, n_tests)}
        ])
    else:
        raise ValueError('extracting Jira is not implemented')

    return df_jira
