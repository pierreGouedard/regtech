# Global import
from typing import Dict, List, Iterable
import pandas as pd
import numpy as np

# Local import
from regtech.datalab.dataops.text.utils import clean_text, levenstein_distance


class FuzzyBagOfWords:

    def __init__(
            self, threshold_inter: float, threshold_levenstein: float, d_stopwords: Dict[str, bool],
            min_len: int = 3
    ):
        self.threshold_inter = threshold_inter
        self.threshold_levenstein = threshold_levenstein
        self.min_len = min_len
        self.stopwords = d_stopwords
        self.map_token2ind = {}

    @staticmethod
    def diff(x: str, y: str) -> int:
        """
        Compute basic character intersection between two strings.

        Parameters
        ----------
        x: str
        y: str

        Returns
        -------
        int
        """
        return max(len(set(x).difference(y)),  len(set(y).difference(x)))

    def fuzzy_token_match(self, tokens: List[str], return_new: bool = True) -> Iterable[str]:
        """
        Fuzzy matching based on levenstein distance.

        Parameters
        ----------
        tokens: list
            list of tokens.
        return_new: bool
            Set whether un mathced tokens should be returned

        Returns
        -------
        Iterable[str]
        """
        for token in tokens:
            is_same = False
            for ref_token in self.map_token2ind.keys():
                n = int((len(token) + len(ref_token)) / 2)

                if self.diff(token[:n], ref_token[:n]) / n < self.threshold_inter:
                    if levenstein_distance(token[:n], ref_token[:n]) / n < self.threshold_levenstein:
                        is_same = True
                        yield ref_token
                        break

            if return_new and not is_same:
                yield token

    def fit(self, df_corpus: pd.DataFrame, l_text_col: List[str]) -> 'FuzzyBagOfWords':
        """
        Fit fuzzy encoder.

        Parameters
        ----------
        df_corpus: DataFrame
            contains corpus to encode.
        l_text_col: list
            List of text columns to encode.

        Returns
        -------
        self
        """
        i = 0
        for _, row in df_corpus.iterrows():
            l_tokens = clean_text('\n'.join(row[l_text_col].tolist()), self.stopwords, min_len=self.min_len)
            for ref_token in self.fuzzy_token_match(l_tokens):
                self.map_token2ind[ref_token] = self.map_token2ind.get(
                    ref_token,  max(self.map_token2ind.values(), default=0) + 1
                )

            if i % 100 == 0:
                print(f'Fitted {i} / {df_corpus.shape[0]} - Dictionnary size is {len(self.map_token2ind)}')

            i += 1

        return self

    def transform(self, df_corpus: pd.DataFrame, l_text_col: List[str]) -> np.ndarray:
        """
        Transform corpus using fitted fuzzy matcher.

        Parameters
        ----------
        df_corpus: DataFrame
            contains corpus to encode.
        l_text_col: list
            List of text columns to encode.

        Returns
        -------
        np.ndarray
            Encoded corpus.
        """
        if not self.map_token2ind:
            raise ValueError('Model FuzzyBagOfWords not fitted')

        # transform corpus
        ax_encoded_tests, i = np.zeros((df_corpus.shape[0], len(self.map_token2ind) + 1)), 0
        for _, row in df_corpus.iterrows():
            # get tokens
            l_tokens = clean_text('\n'.join(row[l_text_col].tolist()), self.stopwords, min_len=self.min_len)

            # Get indices of matched ref token and increment code matrix
            l_indices = [self.map_token2ind[tk] for tk in self.fuzzy_token_match(l_tokens, return_new=False)]
            ax_encoded_tests[i, l_indices] = 1.

            # Increment i
            i += 1

        return ax_encoded_tests

