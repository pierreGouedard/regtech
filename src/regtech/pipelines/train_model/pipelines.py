"""Pipeline for activity pattern."""
# Global import
from kedro.pipeline import Pipeline, node

# Local import
from .commits.load_commits import build_commit_dataset
from .commits.fit_model import fit_feature_model
from .commits.transform_commits import transform_commits
from .tests.encode import encoder_fit_transform
from .tests.build_dataset import build_datasets
from .tests.infer_master_tests import get_master_tests
from .jira.extract_jira import extract_jira_info
from .fit.build_datasets import build_dataset
from .fit.train import train_model
from .fit.evaluate import evaluate


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
                    "project_wrapper": "project_wrapper", "n_component_lficf": "params:n_component_lficf"
                },
                outputs=["transformed_commits", 'lficf_pca_model'],
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
                build_datasets,
                inputs={"path_test": "params:path_test"},
                outputs=["historical_tests", "master_tests", "validation_tests"],
                name="build_test_dataset",
            ),
            node(
                encoder_fit_transform,
                inputs={
                    "df_historical_tests": "historical_tests", "df_master_tests": "master_tests",
                    "df_validation_tests": "validation_tests", "path_stopwords": "params:path_stopwords",
                    "threshold_inter": "params:threshold_inter", "threshold_levenstein": "params:threshold_levenstein"
                },
                outputs=[
                    "encoded_historical_tests", "encoded_master_tests", "encoded_validation_tests", "fuzzy_encoder"
                ],
                name="encode_tests",
            ),
            node(
                get_master_tests,
                inputs={
                    "ax_validation_tests": "encoded_validation_tests", "ax_master_tests": "encoded_master_tests",
                    "df_validation_tests": "validation_tests", "df_master_tests": "master_tests",
                    "d_dict_reg_params": "params:dictionary_regression"
                },
                outputs='scored_validation_tests',
                name="infer_master_tests",
            ),
        ],
        tags=["tests_pipeline"],
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
    return Pipeline(
        [
            node(
                build_dataset,
                inputs={
                    "df_commits": "commit_dataset", "df_jira": "jira_info", "d_feature_commits": "transformed_commits",
                    "n_step_commits": "params:n_step_commits", "n_tests": "params:n_tests"
                },
                outputs=["features", "targets"],
                name="build_dataset",
                tags=["build_dataset"]
            ),
            node(
                train_model,
                inputs={
                    "ax_features": "features", "ax_targets": "targets", "n_kernels": "params:n_kernels",
                    "kernel_size": "params:kernel_size", "learning_rate": "params:learning_rate",
                    "nb_epoch": "params:nb_epoch"
                },
                outputs=["commit2test_network", "commit2test_params"],
                name="train_commit2test",
                tags=["train_commit2test"]
            ),
            node(
                evaluate,
                inputs={
                    "commit2test_network": "commit2test_network", "commit2test_params": "commit2test_params"
                },
                outputs=None,
                name="evaluate_commit2test",
                tags=["evaluate_commit2test"]
            ),
        ],
        tags=["fit_pipeline"],
    )
