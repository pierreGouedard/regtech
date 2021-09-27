# Local import
import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

# Global import


class ProjectWrapper(object):
    """
    Manage the extraction of feature from git history and topology of the project
    """
    def __init__(self, project_dir, l_code_wrapper, multi_project=False):
        """

        Parameters
        ----------
        project_dir
        l_code_wrapper
        multi_project
        """
        if multi_project:
            self.project_dirs = [os.listdir(project_dir)]
        else:
            self.project_dirs = [project_dir]

        self.files = []
        self.code_wrappers = l_code_wrapper
        self.inverse_loc_frequency = None

    def is_file_targeted(self, name: str) -> bool:
        """
        Check whether a file may be processed by a code wrapper

        Parameters
        ----------
        name: str
            File name with extension.

        Returns
        -------
        bool
        """
        return any([w.is_valid_filename(name) for w in self.code_wrappers])

    def extract_file_list(self) -> 'ProjectWrapper':
        """
        Extract the list of file from the root dir.

        Returns
        -------
        self
        """
        for project_dir in self.project_dirs:
            for root, dirs, files in os.walk(project_dir, topdown=True):
                l_files = [os.path.abspath(os.path.join(root, name)) for name in files if self.is_file_targeted(name)]
                self.files.extend(l_files)

        return self

    def get_code_from_pos(self, path_file: str, coord: Tuple[int, int]) -> Optional[Tuple[str, str]]:
        """
        Extract code from file and line coordinate.

        The line coordinate are in the format: (start_line, offset) so thath end_line = start_line + offset

        Parameters
        ----------
        path_file: str
            path of file
        coord: tuple
            coordinate of changes

        Returns
        -------
        tuple
        """

        for wrapper in self.code_wrappers:
            str_code = wrapper.get_code_from_pos(path_file, coord)
            if str_code is not None:
                return wrapper.name, str_code

        return None

    def fit_loc_frequency(self, df_commits: pd.DataFrame) -> None:
        """
        Compute loc frequency over set of commits

        Parameters
        ----------
        df_commits: DataFrame

        Returns
        -------

        """
        self.inverse_loc_frequency = np.zeros(len(self.files))
        for _, df_commit_sub in df_commits.groupby('hash'):
            ax_tmp_freq = np.zeros(self.inverse_loc_frequency.shape)

            for _, row in df_commit_sub.iterrows():
                try:
                    i = self.files.index(row['file_source'])

                except ValueError:
                    i = None

                if i is not None:
                    ax_tmp_freq[i] += 1

            self.inverse_loc_frequency += (ax_tmp_freq > 0).astype(int)

        self.inverse_loc_frequency = np.log(len(df_commits) / (1 + self.inverse_loc_frequency))

    def build_lficf(self, df_commits: pd.DataFrame) -> np.array:
        """
        Loc frequency - Inverse commit frequency transform.

        Each row decribe a code change associated to the the same commit

        Parameters
        ----------
        df_commits: pd.DataFrame

        Returns
        -------
        array

        """
        ax_loc_freq = np.zeros(self.inverse_loc_frequency.shape)

        for _, row in df_commits.iterrows():
            try:
                i = self.files.index(row['file_source'])

            except ValueError:
                i = None

            if i is not None:
                ax_loc_freq[i] += 1

        return ax_loc_freq * self.inverse_loc_frequency





