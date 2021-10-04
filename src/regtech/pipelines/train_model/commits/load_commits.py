import os
from typing import Dict, Any, Optional
import pandas as pd
from random import randint

# Local import
from regtech.datalab.devops.git.history_loader import GitLogLoader


def build_commit_dataset(
        path_project: str, custom_message_regex: Dict[str, str], date_start: str, date_end: str,
        git_options: Dict[str, Any], n_commit_group: Optional[int] = None
) -> pd.DataFrame:
    """
    Load & commits git full history from a github like repository.

    Parameters
    ----------
    path_project: str
        Path of that point to the project.
    custom_message_regex: dict

    date_start: str
        Data after which commits should be considered in the format YYYY-MM-DD.
    date_end: str
        Data until which commits should be considered in the format YYYY-MM-DD.
    git_options: dict
        dict of option to pass to git command line to recover the history

    Returns
    -------
    dict
        Containing key information of git history.
    """
    # Create gt loader and load git log history
    l_commits = []
    for dirname in os.listdir(path_project):
        git_loader = GitLogLoader(
            os.path.join(path_project, dirname), date_start=date_start, date_end=date_end, git_options=git_options,
            custom_message_regex=custom_message_regex
        )
        l_commits.extend(git_loader.extract_git_history())

    if n_commit_group is not None:
        l_commits = [{"cgroup": str(randint(0, n_commit_group - 1)), **d} for d in l_commits]

    print(f'Extraction of {len(l_commits)} code change')

    return pd.DataFrame(l_commits)

