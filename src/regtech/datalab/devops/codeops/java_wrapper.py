# Local import
import os
from typing import List, Tuple, Optional
from pathlib import Path
import re

# Global import


class JavaProjectWrapper(object):
    """"""

    # language meta
    file_extension = '.java'
    name = 'java'

    # List of language keywords
    l_modifiers = ['private', 'default', 'protected', 'public']
    l_modifiers_na = ['static', 'final', 'abstract', 'synchronized', 'volatile', 'transient', 'native']

    # common pattern
    reggr_modifers = rf"(?:(?:{'|'.join(l_modifiers)})\s)?"
    reggr_modifers_na = rf"(?:(?:{'|'.join(l_modifiers_na)})\s)*"

    # class pattern
    reg_class = r"class\s\w+\s\{(?:.|\n|\t|\r|)+\}"
    pattern_class = re.compile("(" + reggr_modifers + reggr_modifers_na + reg_class + ")")

    # Method regex
    reg_method = r".+\s\w+\(.*\)\s(?:throws\s.+\s)?\{(?:.|\n|\t|\r|)+\}"
    pattern_method = re.compile("(" + reggr_modifers + reggr_modifers_na + reg_method + ")")

    def __init__(self, max_offset=75, step=10, max_len_line=100):
        self.max_offset = max_offset
        self.step = step
        self.max_len_line = max_len_line

    @staticmethod
    def filter_groups(match_txt: str, pos_changes: Tuple[int, int], pos_match: Tuple[int, int]) -> bool:
        """

        Parameters
        ----------
        match_txt
        pos_changes
        pos_match

        Returns
        -------

        """

        is_included = pos_match[0] < pos_changes[0] and pos_changes[1] < pos_match[1]
        is_complete = len(re.findall(r'\{', match_txt)) <= len(re.findall(r'\}', match_txt))
        return is_included and is_complete

    @staticmethod
    def line_to_char(l_lines: List[str], line_offset: int, include_last=False) -> int:
        return sum([len(l) for l in l_lines[:line_offset + int(include_last)]])

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
        if os.path.exists(path_file):
            l_lines = Path(path_file).read_text().split('\n')
        else:
            return

        offset, no_match = 5, True
        while no_match and offset < self.max_offset:

            # Get new line coordinates in file
            offset_start, offset_end = min(int(coord[0]), offset), min(len(l_lines) - sum(map(int, coord)), offset)
            start_line_coord, end_line_coord = int(coord[0]) - offset_start, sum(map(int, coord)) + offset
            l_sub_lines = [l[:self.max_len_line] for l in l_lines[start_line_coord:end_line_coord]]

            # Extract code and char coordinate
            code = '\n'.join(l_sub_lines)
            start_char_coord = self.line_to_char(l_sub_lines, offset_start + 1)
            end_char_coord = self.line_to_char(l_sub_lines, offset_start + int(coord[1]) - 1, include_last=True)

            # Get all classes and methods that may be extracted from code and that overlapp with coord of code change
            l_classes = list(filter(
                lambda x: self.filter_groups(x.group(), (start_char_coord, end_char_coord), (x.start(), x.end())),
                [m for m in self.pattern_class.finditer(code)]
            ))
            l_methods = list(filter(
                lambda x: self.filter_groups(x.group(), (start_char_coord, end_char_coord), (x.start(), x.end())),
                [m for m in self.pattern_method.finditer(code)]
            ))

            # Gather code extracted and return it if any
            extracted_code = '\n'.join([x.group() for x in l_classes]) + '\n'.join([x.group() for x in l_methods])
            if extracted_code:
                return extracted_code

            # Increment offset
            offset += self.step

        return None

