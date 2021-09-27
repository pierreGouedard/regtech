from typing import Dict
import numpy as np

# Local import
from regtech.datalab.devops.codeops.project_wrapper import ProjectWrapper
from regtech.datalab.dataops.models.code2vec.code2vec import Code2VecWrapper


def transform_commits(
        df_commits, path_c2v_model: str, project_wrapper: ProjectWrapper
) -> Dict[str, Dict[str, np.array]]:
    """

    Parameters
    ----------
    df_commits
    path_project
    path_c2v_model
    project_wrapper

    Returns
    -------

    """
    # Instantiate c2v model
    code2vec = Code2VecWrapper(model_path=path_c2v_model)

    # Gather code changes
    d_commit_features = {}
    for hash, df_commits_sub in df_commits.groupby('hash'):

        # Compute LFICF features
        ax_lficf = project_wrapper.build_lficf(df_commits_sub)

        l_codes = []
        for _, row in df_commits_sub.iterrows():

            if len({'line_origin+', 'nb_additions', 'file_target'}.intersection(row.index)) < 3:
                continue

            code = project_wrapper.get_code_from_pos(row['file_target'], (row['line_origin+'], row['nb_additions']))
            if code is not None:
                l_codes.append(code)

        # Get vectors from commits
        ax_code_vector = code2vec.predict_code_vector([t[1] for t in l_codes if t is not None])
        if ax_code_vector is not None and ax_lficf is not None:
            d_commit_features[hash] = {'lficf': ax_lficf, 'code_vector': ax_code_vector}

    return d_commit_features






