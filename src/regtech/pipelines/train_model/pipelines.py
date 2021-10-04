"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from .commits.load_commits import build_commit_dataset
from .commits.fit_model import fit_feature_model
from .commits.transform_commits import transform_commits
from .tests.generate_fake_tests import generate_tests
from .tests.extract_test_parameters import extract_test_parameters
from .tests.augment_test_parameters import augment_tests
from .tests.transform_tests import transform_tests
from .jira.extract_jira import extract_jira_info


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


def create_test_pipeline() -> Pipeline:
    """
    Create test pipeline.

    Returns
    -------
    Pipeline

    """
    return Pipeline(
        [
            node(
                generate_tests,
                inputs={"n_tests": "params:n_tests", "l_parameters": "params:test_parameters"},
                outputs="tests",
                name="generate_test_parameters",
                tags=["generate_test_parameters"]
            ),
            node(
                extract_test_parameters,
                inputs={"df_tests": "tests", "regex_parameters": "params:regex_parameters"},
                outputs="test_parameters",
                name="extract_test_parameters",
                tags=["extract_test_parameters"]
            ),
            node(
                augment_tests,
                inputs={"df_tests": 'test_parameters'},
                outputs="augmented_test_parameters",
                name="augment_tests",
                tags=["augment_tests"]
            ),
            node(
                transform_tests,
                inputs={"df_tests": 'augmented_test_parameters'},
                outputs="distributed_tests",
                name="transform_tests",
                tags=["transform_tests"]
            ),
        ],
        tags=["tests_pipeline", "test_augmentation", "test_transform"],
    )


def create_jira_pipeline() -> Pipeline:
    """
    Create jira pipeline.

    Returns
    -------
    Pipeline

    """
    return Pipeline(
        [
            node(
                extract_jira_info,
                inputs={
                    "create_random_join_info": "params:create_random_join_info", "n_tests": "params:n_tests",
                    "n_commit_group": "params:n_commit_group"},
                outputs="jira_info",
                name="extract_jira_info",
                tags=["extract_jira_info"]
            ),
        ],
        tags=["jira_pipeline"],
    )


def create_fit_pipeline() -> Pipeline:
    """
    Create test pipeline.

    Returns
    -------
    Pipeline

    """
    pass
