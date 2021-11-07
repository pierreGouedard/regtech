# Global import
import re
from typing import Union, List, Dict, Optional
from functools import lru_cache
import numpy as np
from pathlib import Path

# Local import and global variable
general_regex = re.compile(r'[aA-zZ]+&[aA-zZ]|[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*|\d{2,}')


@lru_cache()
def levenstein_distance(s: Union[str, List[str]], t: Union[str, List[str]]):
    """
    Recursive implementation of the Levenstein distance between two strings.

    Parameters
    ----------
    s : str
        string 1
    t : str
        string 2

    Returns
    -------
    int
        Levenstein distance between string s and string t.

    """

    # If one of the input word is empty
    if not s:
        return len(t)

    if not t:
        return len(s)

    if s[0] == t[0]:
        return levenstein_distance(s[1:], t[1:])

    l1 = levenstein_distance(s, t[1:])
    l2 = levenstein_distance(s[1:], t)
    l3 = levenstein_distance(s[1:], t[1:])

    return 1 + min(l1, l2, l3)


def relative_hamming_distance(s: str, t: str) -> int:
    min_str = s if len(s) < len(t) else t
    other_str = t if len(s) < len(t) else s
    return sum([x == other_str[i] for i, x in enumerate(min_str)]) / len(min_str)


@lru_cache()
def fast_levenstein_distance(s: Union[str, List[str]], t: Union[str, List[str]], tol: float):
    """
    Recursive implementation of the Levenstein distance that return True if the distance is lower or equal than
    tolerance, False otherwise.

    Parameters
    ----------
    s : str
        String 1.
    t : str
        String 2.
    tol : int
        Tolerance parameter.

    Returns
    -------
    bool
        Indicate whether Levenstein distance between string s and string t is lower or equal than tol, or not.

    """
    # If tol is negative, return false
    if tol < 0:
        return False

    # If one of the input word is empty
    if not s:
        return len(t) <= tol

    if not t:
        return len(s) <= tol

    if s[0] == t[0]:
        return fast_levenstein_distance(s[1:], t[1:], tol=tol)

    return any([
        fast_levenstein_distance(s, t[1:], tol=tol - 1),
        fast_levenstein_distance(s[1:], t, tol=tol - 1),
        fast_levenstein_distance(s[1:], t[1:], tol=tol - 1)
    ])


def lcs(x: Union[str, List[str]], y: Union[str, List[str]]) -> int:
    """
    Compute the maximum subsequence between sequences x and y.

    Parameters
    ----------
    x: list
        First sequence.
    y: list
        Second sequence

    Returns
    -------
    int
    """
    # find the length of the strings
    m, n = len(x), len(y)

    # declaring the array for storing the dp values
    ax_dp = np.zeros((m + 1, n + 1))

    # Core loop
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                ax_dp[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                ax_dp[i, j] = ax_dp[i - 1, j - 1] + 1
            else:
                ax_dp[i, j] = max(ax_dp[i - 1, j], ax_dp[i, j - 1])

    return ax_dp[m, n]


def load_stopwords(path: str) -> Dict[str, bool]:
    return {sw: True for sw in Path(path).read_text().split('\n')}


def clean_text(
        text: str, d_stopwords: Dict[str, bool], l_token_filter: Optional[List[str]] = None, min_len: int = 2
) -> List[str]:
    """
    Clean a text string for NLP analysis.

    Parameters
    ----------
    text: str
        A text (string) to normalize.

    l_token_filter: list of str
        Custom token to filter out.

    Returns
    ----------
    cleaned_text : str
        A text cleaned, needed for transferring text from human language to machine-readable format for further
        processing.

    """
    tokens = tokenize_text_pattern(text.lower(), d_stopwords, min_len)

    if l_token_filter is not None:
        l_token_filter = list(map(lambda x: x.lower(), l_token_filter))
        tokens = [t for t in tokens if t not in l_token_filter]

    return tokens


def tokenize_text_pattern(text: str, d_stopwords: Dict[str, bool], min_len: int) -> List[str]:
    """
    Tokenize text

    Remove campaigns date, seek for <token>x<token> pattern and <c>&<c> patterns using re.pattern technique.

    Parameters
    ----------
    text : str
        text that should be tokenized.

    Returns
    -------
    list
        list of token (str) built from input text.
    """
    # Get token
    other_tokens = [x for x in general_regex.findall(text) if len(x) >= min_len]

    # Remove stopwords
    l_tokens = [w for w in other_tokens if not d_stopwords.get(w, False)]

    return l_tokens