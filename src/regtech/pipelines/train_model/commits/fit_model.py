from typing import List, Dict

# Local import
from regtech.datalab.devops.codeops.java_wrapper import JavaProjectWrapper
from regtech.datalab.devops.codeops.project_wrapper import ProjectWrapper


def fit_feature_model(df_commits, path_project: str) -> ProjectWrapper:
    """
    Load & commits git full history from a github like repository.

    Parameters
    ----------
    path_project: str
        Path of that point to the project.

    Returns
    -------
    dict
        Containing key information of git history.
    """
    project_wrapper = ProjectWrapper(path_project, [JavaProjectWrapper()])
    project_wrapper.extract_file_list()

    # Get location change frequency over commits.
    project_wrapper.fit_loc_frequency(df_commits)

    return project_wrapper

