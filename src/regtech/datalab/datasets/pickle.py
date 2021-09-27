"""Implement Postgres kedro dataset."""
# Global import
from kedro.io.core import AbstractDataSet
from typing import Any, Dict, Optional
import pickle
import dill


class PickleDataSet(AbstractDataSet):
    """``TimescaleDataSet`` load / save data from / to Google Cloud Bucket storage."""
    def __init__(self, path: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new ``GcBucketDataSet``.

        Parameters
        ----------
        path: str
            The path of the bucket query statement.
        parameters: dict
            Provided args that will be used in save or loads in loads or save.

        """
        self.path = path
        default_parameters = {}  # type: Dict[str, Any]

        self.parameters = {**default_parameters, **parameters} if parameters is not None else default_parameters

    def _describe(self) -> Dict[str, Any]:
        """
        Describe the dataset.

        Returns
        -------
        dict
            Dictionary that describes current dataset
        """
        return dict(path=self.path, **self.parameters)

    def _load(self) -> Any:
        """
        Load data from from pickle file.

        Returns
        -------
        Object
            Object loaded from bucket.
        """
        if self.parameters.get('use_dill', False):
            with open(self.path, 'rb') as handle:
                data = pickle.load(handle)
        else:
            with open(self.path, 'rb') as handle:
                data = dill.load(handle)

        return data

    def _save(self, data: Any) -> None:
        """
        Save data to file as pickle.

        Parameters
        ----------
        data: Object
            Object to upload to a bucket.

        Returns
        -------
        None
        """
        if self.parameters.get('use_dill', False):
            with open(self.path, 'wb') as handle:
                pickle.dump(data, handle)
        else:
            with open(self.path, 'wb') as handle:
                dill.dump(data, handle)

    def _exists(self) -> bool:
        """
        Check the existence of data.

        Returns
        -------
        bool
            Indicate whether output exists.

        """
        return False
