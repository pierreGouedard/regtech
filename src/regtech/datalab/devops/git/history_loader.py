# Global import
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import re

# Local import
from regtech.datalab.devops.driver import FileDriver


class GitLogHistoryError(Exception):
    """Raised when shell script to get git log history fails"""
    pass


class GitDiffCommitError(Exception):
    """Raised when shell script to get git commit diff fails"""
    pass


class GitLogLoader(object):
    """
    Manage the extraction of historical git operations from a git repository
    """
    # Commit log key info
    log_history_keys = ['hash', 'owner', 'datetime', 'message']

    # Commit log patterns
    pattern_commit = re.compile(r"(^commit\s(?:(?:[0-9]|[a-z]){40}))")
    pattern_author = re.compile(r"(^Author:*)")
    pattern_mail = re.compile(r"(\<[\w\.-]+@[\w\.-]+\>)")
    pattern_date = re.compile(r'(^Date:\s+)')
    pattern_hour = re.compile(r"([0-9]{2}\:[0-9]{2}:[0-9]{2})")

    # Commit diff key info
    diff_keys = ['file_source', 'file_target', 'line_origin+', 'nb_additions', 'line_origin-', 'nb_deletions']

    # Commit diff patterns
    pattern_diff_file = re.compile(r"(^diff\s\-\-git\s)")
    pattern_diff_pos = re.compile(r"(?:^\@\@\s((?:\-|\+|\,|\s|[0-9])+)\s\@\@)")

    def __init__(
            self, project_path: str, date_start: str, date_end: str, branch: Optional[str] = None,
            git_options: Optional[Dict[str, str]] = None, custom_message_regex: Optional[Dict[str, str]] = None
    ):
        """

        Parameters
        ----------
        project_path: str
            Path of the repo.
        date_start: str
            Date after which commits are loaded
        date_end: str
            Date until which commits are loaded
        branch: str
            branch option (i.e dev..prod)
        git_options: dict
            git option on log and diff details.
        custom_message_regex: dict
            dict of regex if any information needs to be extracted from the commit message

        """
        self.project_path = project_path
        self.branch = branch
        self.date_start = date_start
        self.date_end = date_end
        self.git_options = git_options
        self.custom_message_regex = custom_message_regex or {}

    @staticmethod
    def extract_log_info(key: str, line: str) -> Tuple[Optional[str], bool]:
        """

        Parameters
        ----------
        key: str
            Key referring to the information it  seeks.
        line: str
            line of the git log

        Returns
        -------
        tuple
            The line if found and a boolean.

        """
        if key == 'hash':
            if GitLogLoader.pattern_commit.match(line) is not None:
                return line.split('commit ')[1], True

        elif key == 'owner':
            if GitLogLoader.pattern_author.match(line) and len(GitLogLoader.pattern_mail.findall(line)) == 1:
                return GitLogLoader.pattern_mail.findall(line)[0], False

        elif key == 'datetime':
            if GitLogLoader.pattern_date.match(line) and len(GitLogLoader.pattern_hour.findall(line)) == 1:
                return GitLogLoader.pattern_date.sub('', line), False

        elif key == 'message':
            return line, False

        else:
            raise ValueError(f'key not knowns {key}')

        return None, False

    @staticmethod
    def format_commit_path(project_dir: str, commit_path: str) -> str:
        """
        Get absolute path.

        Parameters
        ----------
        project_dir
        commit_path

        Returns
        -------

        """
        return os.path.abspath(os.path.join(project_dir, commit_path[2:]))

    @staticmethod
    def process_git_message(l_messages: List[str], d_regex: Dict[str, str]) -> Dict[str, str]:
        """
        Find specific information in the commit message;

        Parameters
        ----------
        l_messages: list
            List of commits message
        d_regex: dict
            dict of regex

        Returns
        -------
        dict
        """
        d_message_info, text = {}, " ".join(l_messages)
        for k, regex in d_regex.items():
            d_message_info[k] = ','.join(re.findall(regex, text))
            text = re.sub(regex, '', text)

        d_message_info['message'] = text

        return d_message_info

    @staticmethod
    def is_diff_key_complete(d_diff_keys: Dict[str, str]) -> bool:
        """
        Check whether dict of commit diff info is filled with all necessary keys.

        Args:
            d_diff_keys: dict
                Dictionnary containing commit diff informations.

        Returns:
            bool

        """
        return all([k in d_diff_keys.keys() for k in GitLogLoader.diff_keys])

    def extract_git_history(self) -> List[Dict[str, str]]:
        """
        Main method to extract commit log and commit diff info.

        Returns
        -------
        list

        """
        # Get and parse log history of commits
        commit_history = self.extract_commit_log()
        l_commits = self.parse_log_history(commit_history)

        # Get commit diff information
        l_commits = self.extract_commit_diff(l_commits)

        return l_commits

    def extract_commit_log(self) -> str:
        """
        Extract log history via bash command 'git log'.

        Returns
        -------
        str
        """
        # Create tmp file to receive log history
        driver = FileDriver('tmp_file', 'Git log history file driver')
        tmp_file = driver.TempFile(prefix="tmp_", suffix=".txt")

        # Use shell subprocess to load git history in tmp file
        shell_script = f"""cd {os.path.abspath(self.project_path)} && \
        git log {" ".join(self.git_options['log'])} --after={self.date_start} --until={self.date_end} >> {tmp_file.path}
        """
        return_code = os.system(shell_script)

        if return_code != 0:
            raise GitLogHistoryError('Git log history failed to be loaded')

        history = Path(tmp_file.path).read_text()
        tmp_file.remove()

        return history

    def parse_log_history(self, commit_history: str) -> List[Dict[str, str]]:
        """
        Parse commit logs.

        Parameters
        ----------
        commit_history: str
            result of bash command 'git log'

        Returns
        -------
        list

        """
        l_commits, end_commit, d_commit_info, l_messages = [], False, {}, []
        for line in commit_history.split('\n'):
            for k in self.log_history_keys:
                res, end_commit = self.extract_log_info(k, line)

                if res is None:
                    continue

                if end_commit:
                    # Get message information
                    d_message_info = self.process_git_message(l_messages, self.custom_message_regex)

                    # Gather info
                    l_commits.append({**d_message_info, **d_commit_info.copy()})
                    d_commit_info, end_commit, l_messages = {k: res}, False, []
                    break

                elif k == 'message':
                    l_messages.append(res)
                    break

                else:
                    d_commit_info[k] = res
                    break

        # return parsed commmit minus first element of list
        return l_commits[1:]

    def extract_commit_diff(self, l_commits: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Extract commit diff info.

        the information that are targeted is the source and target file and the position of change in the file.

        Parameters
        ----------
        l_commits: dict
            list of dict with commit information and a key 'hash' from which the diff can be retrieved.

        Returns
        -------
        dict

        """
        l_new_commits = []
        for d_commit_info in l_commits:
            # Create tmp file to receive log history
            driver = FileDriver('tmp_file', 'Git commit diff file driver')
            tmp_file = driver.TempFile(prefix="tmp_", suffix=".txt")

            # Use shell subprocess to load git history in tmp file
            shell_script = f"""cd {os.path.abspath(self.project_path)} && \
            git show {d_commit_info['hash']} {" ".join(self.git_options['diff'])} >> {tmp_file.path}
            """
            return_code = os.system(shell_script)

            if return_code != 0:
                raise GitDiffCommitError('Git log history failed to be loaded')

            commit_diff = Path(tmp_file.path).read_text()

            # Parse diff
            l_new_commits.extend([{**d_commit_info, **d} for d in self.parse_commit_diff(commit_diff)])

        return l_new_commits

    def parse_commit_diff(self, commit_diff: str) -> List[Dict[str, str]]:
        """
        Parse commit diff.

        Parameters
        ----------
        commit_diff: str

        Returns
        -------
        list
        """
        def split_pos(x: str, i: int):
            """
            Split positions of changes.

            It expect a git formating of position i.e @@ -20,7 +10,10 @@.

            Parameters
            ----------
            x: str
                Git formatted position of changes in text.
            i: int
                Index of position (0: starting line, 1: offset)

            Returns
            -------
            str
                Position param of changes
            """
            return re.findall(r"([0-9]+)", x)[i]

        l_diff_info, l_sub_diff_info, d_diff_info = [], [], {}
        for line in commit_diff.split('\n'):
            if self.pattern_diff_file.match(line):

                # Changing source / target file
                l_diff_info.extend([d.copy() for d in l_sub_diff_info if self.is_diff_key_complete(d)])
                d_diff_info, l_sub_diff_info = {}, []

                # Extract file paths
                files = self.pattern_diff_file.sub('', line)
                d_diff_info['file_source'] = self.format_commit_path(self.project_path, files.split(' ')[0])
                d_diff_info['file_target'] = self.format_commit_path(self.project_path, files.split(' ')[1])

            elif self.pattern_diff_pos.match(line):
                # Extract addition and deletion pos
                [del_pos_info, add_pos_info] = self.pattern_diff_pos.findall(line)[0].split(' ')

                # Update dict of info
                try:
                    d_diff_info.update({
                        'line_origin-': split_pos(del_pos_info, 0), 'nb_deletions': split_pos(del_pos_info, 1),
                        'line_origin+': split_pos(add_pos_info, 0), 'nb_additions': split_pos(add_pos_info, 1)
                    })
                except IndexError:
                    continue

            # Break if all info has been recovered
            if self.is_diff_key_complete(d_diff_info):
                l_sub_diff_info.append(d_diff_info.copy())
                d_diff_info = {'file_source': d_diff_info['file_source'], 'file_target': d_diff_info['file_target']}

        return l_diff_info
