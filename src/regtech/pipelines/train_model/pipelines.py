"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from .commits.load_commits import build_commit_dataset
from .commits.fit_model import fit_feature_model
from .commits.transform_commits import transform_commits


def create_commit_pipeline() -> Pipeline:
    """
    Create commit pipeline.

    Returns
    -------
    Pipeline

    """
    return Pipeline(
        [
            node(
                build_commit_dataset,
                inputs={
                    "path_project": "params:path_project", "custom_message_regex": "params:custom_message_regex",
                    "date_start": "params:date_start", "date_end": "params:date_end",
                    "git_options": "params:git_options"
                },
                outputs="commit_dataset",
                name="load_and_process_git",
                tags=["load_and_process_git"]
            ),
            node(
                fit_feature_model,
                inputs={"df_commits": 'commit_dataset', "path_project": "params:path_project"},
                outputs="project_wrapper",
                name="fit_feature_model",
                tags=["fit_feature_model"]
            ),
            node(
                transform_commits,
                inputs={
                    "df_commits": 'commit_dataset', "path_c2v_model": "params:path_c2v",
                    "project_wrapper": "project_wrapper"
                },
                outputs="transformed_commits",
                name="transform_commits",
                tags=["transform_commits"]
            ),
        ],
        tags=["train_model", "git", "code_wrapper", "commit_to_features", "load_and_process"],
    )


def create_jira_pipeline() -> Pipeline:
    """
    Create jira pipeline.

    Returns
    -------
    Pipeline

    """
    pass


def create_defect_pipeline() -> Pipeline:
    """
    Create defect pipeline.

    Returns
    -------
    Pipeline

    """
    pass


def create_test_pipeline() -> Pipeline:
    """
    Create test pipeline.

    Returns
    -------
    Pipeline

    """
    pass


def create_train_pipeline() -> Pipeline:
    """
    Create test pipeline.

    Returns
    -------
    Pipeline

    """
    pass
