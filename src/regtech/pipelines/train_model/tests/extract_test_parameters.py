# Global import
from typing import Optional, Dict
import pandas as pd
from faker import Faker
from random import randint
import re
import numpy as np

# Local import
from regtech.datalab.dataops.encoders.param_encoder import ParamEncoder


def extract_test_parameters(
        df_tests: pd.DataFrame, regex_parameters: Optional[re.Pattern] = None
) -> Dict[int, np.ndarray]:
    """

    Args:
        df_tests:
        regex_parameters

    Returns:

    """
    # Extract parameters
    param_encoder = ParamEncoder(re.compile(regex_parameters))
    param_encoder.fit(df_tests)
    d_encoded_tests = param_encoder.transform(df_tests)

    return d_encoded_tests



