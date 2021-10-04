# Global import
import pandas as pd


def augment_tests(df_tests: pd.DataFrame) -> pd.DataFrame:
    print('Currently not implemented')
    return df_tests


# Global import
from typing import Optional, List
import pandas as pd
from faker import Faker
from random import randint
import re
import numpy as np

# Local import
from regtech.datalab.dataops.encoders.param_encoder import ParamEncoder


def generate_tests(n_tests: Optional[int] = None, l_parameters: Optional[List[str]] = None) -> pd.DataFrame:
    """

    Args:
        create_random_test:
        n_tests:
        l_parameters:
        regex_parameters

    Returns:

    """
    # Build fake test data
    fake = Faker()
    l_words = l_parameters + fake.sentence(nb_words=20).split(' ')
    df_tests = pd.DataFrame([
        {
            "id": i,
            'desc': fake.sentence(ext_word_list=l_words, nb_words=30).replace('<int>', str(randint(0, 100)))
        } for i in range(n_tests)
    ])

    return df_tests



