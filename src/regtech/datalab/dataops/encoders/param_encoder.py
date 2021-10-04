# Global import
from typing import Dict
import pandas as pd
import re
import numpy as np
from collections import deque


class ParamEncoder:

    pattern_int = re.compile(r'[0-9]+')

    def __init__(self, pattern_param: re.Pattern, desc_col='desc'):
        self.desc_col = desc_col

        self.pattern_param = pattern_param
        self.cat_param_map, self.num_param_map = {}, {}

    def update_param_map(self, x: str, ptype: str):
        is_existing = x in self.cat_param_map.values() or x in self.num_param_map.values()

        if not is_existing:

            k = max(max(self.cat_param_map.keys() or [-1]), max(self.num_param_map.keys() or [-1]))
            if ptype == 'cat':
                self.cat_param_map.update({k + 1: x})
            elif ptype == 'num':
                self.num_param_map.update({k + 1: x})

    def fit(self, df):
        for x in df[self.desc_col]:
            # Get categorical params
            deque(map(lambda x: self.update_param_map(x, 'cat'), [
                r'='.join([x[0], x[1]]) for x in self.pattern_param.findall(x) if self.pattern_int.match(x[1]) is None
            ]))

            # Get nuerical params
            deque(map(lambda x: self.update_param_map(x, 'num'), [
                rf'{x[0]}=([0-9]+)' for x in self.pattern_param.findall(x) if self.pattern_int.match(x[1]) is not None
            ]))

    def transform(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """

        Args:
            df:

        Returns:

        """
        n, d_encoded = max(max(self.cat_param_map.keys() or [-1]), max(self.num_param_map.keys() or [-1])) + 1, {}
        for i, x in df.iterrows():
            ax_encoded = np.zeros(n)
            for ind, p in self.cat_param_map.items():
                if re.findall(p, x[self.desc_col]):
                    ax_encoded[ind] = 1.

            for ind, p in self.num_param_map.items():
                if re.findall(p, x[self.desc_col]):
                    ax_encoded[ind] = float(re.findall(p, x[self.desc_col])[0])

            d_encoded[int(i)] = ax_encoded

        return d_encoded

