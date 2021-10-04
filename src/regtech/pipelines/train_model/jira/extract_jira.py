from typing import Dict, Any, Optional
import pandas as pd


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
        # TODO: generate fake join table between test
        df_jira = pd.DataFrame()
    else:
        raise ValueError('extracting Jira is not implemented')

