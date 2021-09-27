"""Implement Postgres kedro dataset."""
# Global import
from kedro.io.core import AbstractDataSet
from typing import Any, Dict, Optional
import os
from tensorflow.keras.models import load_model, Model


class KerasModelDataSet(AbstractDataSet):
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

    def _load(self) -> Model:
        """
        Load data from from pickle file.

        Returns
        -------
        Object
            Object loaded from bucket.
        """
        custom_objects = None
        if self.parameters.get('custom_object_module', None) is not None:
            l_classes = [c for c in self.parameters['custom_classes'] if c is not None]
            module = __import__(self.parameters['custom_object_module'], fromlist=l_classes)
            custom_objects = {c: getattr(module, c) for c in l_classes}

        model = load_model(self.path, custom_objects=custom_objects)
        return model

    def _save(self, model: Model) -> None:
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
        model.save(self.path)

    def _exists(self) -> bool:
        """
        Check the existence of data.

        Returns
        -------
        bool
            Indicate whether output exists.

        """
        return False
