# Local import
import os
from typing import List, Tuple, Optional
from pathlib import Path
import re
import multiprocessing

# Global import
import IPython


class JavaProjectWrapper(object):
    """"""

    # language meta
    file_extension = '.java'
    name = 'java'

    # other pattern to exclude
    pattern_other = re.compile('((?:if|else|while|for|do|switch)\s\(.+\)\s\{)')

    # List of language keywords
    l_modifiers = ['private', 'default', 'protected', 'public']
    l_modifiers_na = ['static', 'final', 'abstract', 'synchronized', 'volatile', 'transient', 'native']

    # common pattern
    reggr_modifers = rf"(?:(?:{'|'.join(l_modifiers)})\s)?"
    reggr_modifers_na = rf"(?:(?:{'|'.join(l_modifiers_na)})\s){{0,3}}"

    # class pattern
    reg_class = r"class\s\w{0,100}\s\{"
    pattern_class = re.compile("(" + reggr_modifers + reggr_modifers_na + reg_class + ")")

    # Method regex
    reg_method = r".{0,100}\s\w{0,50}\(.{0,300}\)\s(?:throws\s.{0,200}\s)?\{"
    pattern_method = re.compile("(" + reggr_modifers + reggr_modifers_na + reg_method + ")")

    def __init__(self, max_offset=75, step=10, max_len_line=300):
        self.max_offset = max_offset
        self.step = step
        self.max_len_line = max_len_line

    @staticmethod
    def line_to_char(l_lines: List[str], line_offset: int, include_last=False) -> int:
        return sum([len(l) for l in l_lines[:line_offset + int(include_last)]])

    @staticmethod
    def relative_position(coord_left: Tuple[int, int], coord_right: Tuple[int, int]) -> Tuple[bool, bool]:
        """
        Check whether 1D coord left is overlapping or included into 1D coord right.
        Args:
            coord_left: tuple
                Tuple with start and end value of the 1D range of left object.
            coord_right: tuple
                Tuple with start and end value of the 1D range of left object.

        Returns:
        tuple
            first boolean indicate is_overlapping, second i_included
        """

        if coord_right[0] < coord_left[0]:
            if coord_right[1] >= coord_left[1]:
                return True, True

            elif coord_right[1] > coord_left[0]:
                return True, False

        elif coord_right[1] < coord_left[1]:
            return True, False

        return False, False

    @staticmethod
    def is_valid_match(match_txt: str) -> bool:
        """
        Check if the number of { is equal to the number of }.

        Parameters
        ----------
        match_txt: str

        Returns
        -------
        tuple
        """
        return len(re.findall(r'\{', match_txt)) == len(re.findall(r'\}', match_txt))

    @staticmethod
    def extract_class(l_lines: List[str]) -> List[str]:
        """

        Args:
            l_lines:

        Returns:

        """
        l_candidate_end_lines = [i for i in range(len(l_lines)) if re.findall(r'\}', l_lines[i])]

        for n in l_candidate_end_lines:
            if JavaProjectWrapper.is_valid_match('\n'.join(l_lines[:n + 1])):
                return l_lines[:n + 1]

    def is_valid_filename(self, name):
        return os.path.splitext(name)[1] == self.file_extension

    def is_valid_path(self, path):
        return os.path.isfile(path) and os.path.splitext(path)[1] == self.file_extension

    def get_code_from_pos(self, path_file: str, coord: Tuple[int, int]) -> Optional[str]:
        """
        Extract code from line coordinates and file path

        Parameters
        ----------
        path_file
        coord

        Returns
        -------

        """
        if os.path.exists(path_file) and self.is_valid_filename(path_file):
            l_lines = Path(path_file).read_text().split('\n')
        else:
            return

        if len(l_lines) < sum(coord) - 1:
            return

        # Get class or method before change that overlap with code change coord
        l_code, n, offset = [], int(coord[0]), 0
        while n >= 0:
            # Find class or method prototype
            match_class, match_method = self.pattern_class.findall(l_lines[n]), self.pattern_method.findall(l_lines[n])
            other_match = self.pattern_other.findall(l_lines[n])

            if not match_method and not match_class or other_match:
                offset += 1
                n = int(coord[0]) - offset
                continue

            # Find closing bracket of the class / method and test for relative position with code change
            l_sub_lines = self.extract_class(l_lines[n:])
            is_overlap, is_contained = self.relative_position(coord, (n, n + len(l_sub_lines)))

            if is_contained:
                return '\n'.join([l for l in l_sub_lines if l])

            elif is_overlap:
                l_code.append('\n'.join([l for l in l_sub_lines if l]))
                break

            offset += 1
            n = int(coord[0]) - offset

        # Get class or method after change that overlap with code change coord
        n, n_end, offset = coord[0], sum(coord), 0
        while n <= n_end:
            # Find class or method prototype
            match_class, match_method = self.pattern_class.match(l_lines[n]), self.pattern_method.match(l_lines[n])
            other_match = self.pattern_other.findall(l_lines[n])

            if not match_method and not match_class or other_match:
                offset += 1
                n = int(coord[0]) + offset
                continue

            # Find closing bracket of the class / method and test for relative position with code change
            l_sub_lines = self.extract_class(l_lines[n:])
            is_overlap, _ = self.relative_position(coord, (n, n + len(l_sub_lines)))

            if is_overlap:
                l_code.append('\n'.join([l for l in l_sub_lines if l]))
                break

            offset += 1
            n = int(coord[0]) + offset

        if l_code:
            return '\n'.join(l_code)

        return
