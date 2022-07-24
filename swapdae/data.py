import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def swap_dataframe(input_df, prob=0.15):
    total_rows = len(input_df)
    n_rows = int(round(total_rows * prob))
    n_cols = len(input_df.columns)

    def _gen_indices(_total_rows, _n_rows, _n_cols):
        _rows = np.random.randint(0, _total_rows, size=(_n_rows, _n_cols))
        _cols = np.repeat(
            np.arange(_n_cols).reshape(1, -1), repeats=_n_rows, axis=0)
        return _rows, _cols

    rows, cols = _gen_indices(total_rows, n_rows, n_cols)
    swap_data = input_df.values
    to_place = swap_data[rows, cols]

    rows, cols = _gen_indices(total_rows, n_rows, n_cols)
    swap_data[rows, cols] = to_place

    # output result
    dtypes = {col: typ for col, typ in zip(input_df.columns, input_df.dtypes)}
    swap_df = pd.DataFrame(columns=input_df.columns, data=swap_data)
    swap_df = swap_df.astype(dtypes, copy=False)

    return swap_df


class SwapNoiseDataset(Dataset):
    def __init__(self, raw_df, swap_prob=0.15):
        super(SwapNoiseDataset, self).__init__()
        self.swap_prob = swap_prob
        self._target_df = raw_df
        self._swap_df = swap_dataframe(copy.deepcopy(raw_df), self.swap_prob)

    def swap(self):
        self._swap_df = swap_dataframe(copy.deepcopy(self._target_df), self.swap_prob)
        return None

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, i):
        return self._swap_df.iloc[i, :].values.astype(np.double), \
            self._target_df.iloc[i, :].values.astype(np.double)


__all__ = [
    'SwapNoiseDataset'
]
