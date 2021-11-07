# global import
from sklearn.linear_model import Lasso
from typing import Dict, Any
import numpy as np
import pandas as pd
import copy


class LassoDictRegression:
    alpha_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    def __init__(self, params_lasso: Dict[str, Any], ax_dict: np.ndarray):
        self.model = Lasso(**{**params_lasso, **{'positive': True}})
        self.dictionary = ax_dict
        self.coefs = None

    def set_alpha(self, alpha):
        """

        Parameters
        ----------
        alpha

        Returns
        -------

        """
        self.model.alpha = alpha

    def describe_alpha(self, ax_mixtures: np.ndarray) -> Dict[str, Any]:
        """

        Parameters
        ----------
        ax_mixtures

        Returns
        -------

        """
        d_info = {f'alpha={k}': {} for k in self.alpha_values}
        for alpha in self.alpha_values:
            model, d_info_sub = copy.deepcopy(self.model), {}
            model.alpha = alpha

            for ax_mixture in ax_mixtures:
                # Fit model
                model.fit(self.dictionary, ax_mixture)

                # Get score
                d_info_sub['score'] = d_info_sub.get('score', 0) + model.score(self.dictionary, ax_mixture)
                d_info_sub['nnz'] = d_info_sub.get('nnz', 0) + (model.coef_ > 1e-6).sum()

            d_info[f'alpha={alpha}'] = {k: v / ax_mixtures.shape[0] for k, v in d_info_sub.items()}

        return d_info

    def infer_coefs(self, ax_mixture: np.ndarray) -> 'LassoDictRegression':
        """

        Parameters
        ----------
        ax_mixtures

        Returns
        -------

        """
        self.model.fit(self.dictionary, ax_mixture)
        return self

    def score_atoms(self, df_atoms: pd.DataFrame, index_col: str, sum_to_one: bool = True) -> pd.DataFrame:
        """

        Parameters
        ----------
        df_atoms
        index_col

        Returns
        -------

        """
        df_coefs = pd.DataFrame(self.model.coef_[:, np.newaxis], columns=['score']).assign(**{
            index_col: np.arange(self.model.coef_.shape[0]).astype(int),
            'score': lambda x: x.score if not sum_to_one else x.score / x.score.sum()
        })
        return df_atoms.merge(df_coefs, on=index_col, how='left')
