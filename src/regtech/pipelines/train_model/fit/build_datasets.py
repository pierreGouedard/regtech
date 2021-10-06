# Global import
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import time

# Local import


def build_dataset(
        df_commits: pd.DataFrame, df_jira: pd.DataFrame, d_feature_commits: Dict[str,Dict[str, np.ndarray]],
        n_step_commits: int, n_tests: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset.

    Args:
        df_commits: DataFrame
            tabular information about commit.
        df_jira: DataFrame
            Jira information to link commits to tests.
        d_feature_commits: dict
            dict contaning commit's features.
        n_step_commits: int
            Max number of commits considered.
        n_tests: number of 'center' tests whose activation should be predicted.

    Returns:
    tuple
        features and targets as numpy arrays.
    """

    # Get all hash that has features
    l_hash = list(d_feature_commits.keys())

    # Gather commits by issue key and merge with jira
    df_commits = df_commits.loc[df_commits.hash.isin(l_hash), ['line_origin+', 'nb_additions', 'nb_deletions', 'hash']]\
        .groupby('hash')\
        .agg({'line_origin+': min, 'nb_additions': sum, 'nb_deletions': sum})\
        .reset_index()\
        .assign(commit_group=df_commits.hash.apply(lambda x: np.random.choice(df_jira.commit_group.unique())))\

    # Build features and target
    l_features, l_targets = [], []
    for cgid, df_sub in df_commits.groupby('commit_group'):

        # Gather test features
        ax_target = np.zeros(n_tests)
        for tid in df_jira.loc[df_jira.commit_group == cgid].test_id.unique():
            ax_target[tid] = 1
        l_targets.append(ax_target.copy())

        # Gather commit features
        ax_features = pad_commit_features(list(df_sub.hash.unique()), d_feature_commits, n_step_commits)
        l_features.append(ax_features.copy())

    return np.stack(l_features), np.stack(l_targets)


def pad_commit_features(
        l_hash: List[str], d_feature_commits: Dict[str,Dict[str, np.ndarray]], n_step_commits: int
) -> np.ndarray:
    """
    Pad features commit with 0 so that every commit group feature have same dim.

    Parameters
    ----------
    l_hash: list
        List of hash belonging to a given group.
    d_feature_commits: dict
        Commit features keys are the hash of the commits.
    n_step_commits: int
        Max number of commits considered.

    Returns
    -------

    """
    l_feature_group = []
    for i, hash in enumerate(l_hash):

        d_feature = d_feature_commits[hash]
        l_feature_group.append(np.hstack([d_feature['code_vector'], d_feature['lficf']]))

    if len(l_feature_group) >= n_step_commits:
        return np.stack(l_feature_group[:n_step_commits])

    else:
        return np.vstack([
            np.stack(l_feature_group), np.zeros((n_step_commits - len(l_feature_group), l_feature_group[0].shape[0]))
        ])
