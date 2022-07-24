import os
import sys
import copy
import torch
import logging
import traceback
import numpy as np
import pandas as pd 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List, Dict
from collections import defaultdict


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return None


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    return None


def load_weights(model, weights_path):
    try:
        model.load_state_dict(
            torch.load(weights_path)
        )

        logging.info(
            '[Load Weights] Successfully load weights from `{}`.'.format(weights_path)
        )

    except Exception as e:
        logging.warning('[Load Weights] Failed to load weight from {}'.format(weights_path))
        logging.debug(e)

    return None


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return None


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True
    return None


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def show_message(message: str, message_level: int, verbose: int) -> None:
    if verbose >= message_level:
        print(message)
    return None 


class Meter:
    def __init__(self) -> None:
        self.reset() 

    @property
    def names(self) -> List:
        return [k for k in self._history.keys()]

    @property
    def history(self) -> Dict:
        return self._history

    def __getitem__(self, name: str) -> dict:
        return self._history[name]

    def update(self, updates: Dict) -> 'Meter':
        for key, val in updates.items():
            self._history[key].append(val)
        return self

    def reset(self) -> None:
        self._history = defaultdict(list)
        return None 

    def merge(self, meter) -> None:
        if len(self._history) == 0:
            self._history = meter._history
        else:
            for key, val in self._history.items():
                val += meter._history[key]
        return 

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)


class MultiColumnLabelEncoder:
    """
    Apply label encoding on multiple columns.

    Example:
        .. test code::

        data = np.array([[3.2, 0.2, 'A', 0.2, 0.2, 0.2, 0.2, 122, 'W', 'PP'],
                         [2.4, 0.6, 'A', 0.6, 0.6, 0.6, 0.6, 156, 'Q', 'QA'],
                         [0.3, 0.1, 'B', 0.1, 0.1, 0.1, 0.1, 111, 'Q', 'FF'],
                         [1.8, 0.0, 'C', 0.0, 0.0, 0.0, 0.0, 100, 'Q', 'FF'],
                         [0.7, 1.1, 'B', 1.1, 1.1, 1.1, 1.1, 101, 'W', 'QA']], dtype=object)
        
        encoder = MultiColumnLabelEncoder(indices=[2, 8, 9])
        encoded_data = encoder.fit_transform(data)

        assert encoded_data == np.array([[3.2, 0.2, 0, 0.2, 0.2, 0.2, 0.2, 122, 1, 1],
                                         [2.4, 0.6, 0, 0.6, 0.6, 0.6, 0.6, 156, 0, 2],
                                         [0.3, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 111, 0, 0],
                                         [1.8, 0.0, 2, 0.0, 0.0, 0.0, 0.0, 100, 0, 0],
                                         [0.7, 1.1, 1, 1.1, 1.1, 1.1, 1.1, 101, 1, 2]], dtype=object)

    """
    def __init__(self, indices: List) -> None:
        self.indices = indices
        self.encoders = {}

    def fit(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Fit Labelencoder on specified columns.

        Args:
            data (pd.DataFrame or np.ndarray):
                Input table with shape = (sample_size, num_columns)

        Returns:
            self

        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        for index in self.indices:
            self.encoders[index] = LabelEncoder().fit(data[:, index])

        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Apply data transformation.

        Args:
            data (pd.DataFrame or np.ndarray):
                Input table with shape = (sample_size, num_columns)

        Returns:
            data (np.ndarray):
                Processed data with encoded features. 

        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if len(self.encoders) == 0:
            raise RuntimeError('This `DataGrouper` instance is not fitted yet.')

        for index in self.indices:
            data[:, index] = self.encoders[index].transform(data[:, index])

        return data

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Combine `fit` and `transform` methods

        Args:
            data (pd.DataFrame or np.ndarray):
                Input data with shape = (sample_size, num_columns)

        Returns:
            data (np.ndarray):
                Processed data with encoded features. 

        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        for index in self.indices:
            self.encoders[index] = LabelEncoder().fit(data[:, index])
            data[:, index] = self.encoders[index].transform(data[:, index])
        
        return data

    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform labels back to original encoding.

        Args:
            data (np.ndarray):
                Data contains encoded labels with shape = (sample_size, num_columns)
        
        Returns:
            data (np.ndarray):
                Data with original encoding.

        """
        if len(self.encoders) == 0:
            raise RuntimeError('This `DataGrouper` instance is not fitted yet.')
        
        for index in self.indices:
            data[:, index] = self.encoders[index].inverse_transform(data[:, index].astype(int))

        return data

    @property
    def cardinalities(self):
        return [len(le.classes_) for _, le in self.encoders.items()]

    def get_cardinalities(self, index):
        le = self.encoders.get(index)

        if le is not None:
            return len(le.classes_) 
        else:
            raise IndexError


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        num_indices: List[int], 
        cate_indices: List[int] = None, 
        numerical_scaler=None
    ) -> None:
        self.num_indices = np.array(num_indices)
        self.cate_indices = cate_indices
        self.scaler = StandardScaler() \
            if numerical_scaler is None else numerical_scaler
        self.label_encoder = MultiColumnLabelEncoder(self.cate_indices) \
            if cate_indices is not None else None
        
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.DataFrame, pd.Series, np.ndarray] = None
    ) -> 'Preprocessor':
        if not isinstance(X, (pd.DataFrame, np.ndarray)): 
            raise TypeError('Invalid input type.')

        if y is not None and not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError('Invalid input type.')
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        self.scaler.fit(X[:, self.num_indices])

        if self.label_encoder is not None:
            self.label_encoder.fit(X)

        return self
    
    def transform(
        self, 
        X, 
        y=None
    ) -> np.ndarray:
        if not isinstance(X, (pd.DataFrame, np.ndarray)): 
            raise TypeError('Invalid input type.')

        if y is not None and not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError('Invalid input type.')
        
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_trans = copy.deepcopy(X)
        X_trans[:, self.num_indices] = self.scaler.transform(X_trans[:, self.num_indices])

        if self.label_encoder is not None:
            X_trans = self.label_encoder.transform(X_trans)

        return X_trans


__all__ = [
    'DataPreprocessor', 'MultiColumnLabelEncoder',
]
