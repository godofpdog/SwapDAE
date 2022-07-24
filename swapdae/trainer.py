import os 
import copy
import torch
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from .utils import Meter, show_message
from .model import AutoEncoder
from .data import SwapNoiseDataset
from typing import Union, List, Dict


class SwapDAEPretrainer:
    def __init__(
        self, 
        batch_size: int = 256,
        max_epochs: int = 100,
        swap_prob: float = 0.15,
        verbose: int = 1,
        save_path: str = None,
        device: str = 'cuda'
    ) -> None:
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.swap_prob = swap_prob
        self.verbose = verbose
        self.device = device
        self.save_path = '' if save_path is None else save_path
        self._optimizer = None 
        self._schedulers = None
        self._step_meter = Meter()
        self._epoch_meter = Meter() 

    def register_optimizer(self, optimizer) -> 'SwapDAEPretrainer':
        if not isinstance(optimizer, Optimizer):
            raise TypeError('Invalid optimizer')
        self._optimizer = optimizer 
        return self

    def register_schedulers(self, schedulers) -> 'SwapDAEPretrainer':
        _schedulers = []

        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for scheduler in schedulers:
            if not isinstance(scheduler, (torch.optim.lr_scheduler._LRScheduler, EarlyStopper)):
                raise TypeError('Invalid scheduler.')
            _schedulers.append(scheduler)
        self._schedulers = _schedulers
        return self

    def fit(self, model: AutoEncoder, dataset: Union[np.ndarray, pd.DataFrame]) -> None:
        if not isinstance(model, AutoEncoder):
            raise TypeError('Input argument ``model`` must be an ``AutoEncoder`` instance.')

        if not isinstance(dataset, (np.ndarray, pd.DataFrame)):
            raise TypeError('Input argument ``dataset`` must be an ``pandas.DataFrame`` or ``numpy.ndarray`` instance.')

        dataset = pd.DataFrame(dataset) if isinstance(dataset, np.ndarray) else dataset
        dataset = SwapNoiseDataset(copy.deepcopy(dataset), swap_prob=self.swap_prob)

        self.on_training_begin(model)

        for epoch in range(self.max_epochs):
            loss = self.train_epoch(model, dataset)
            show_message('[Epoch][{}][Loss][{}]'.format(epoch + 1, loss), 1, self.verbose)
            self._epoch_meter.update({'loss': loss})

            if self._schedulers is not None:
                for scheduler in self._schedulers:
                    scheduler.step()

        show_message('Training complete.', 1, self.verbose)

        try:
            weights_path = os.path.join(self.save_path, 'weights.pt')
            torch.save(model.state_dict(), weights_path)
            show_message('Successfully save weights to {}.'.format(weights_path), 1, self.verbose)
        except Exception as e:
            show_message('Failed to save weights.', 1, self.verbose)
            show_message('{}'.format(e), 1, self.verbose)

        return None
            
    def on_training_begin(self, model):
        model.to(self.device)
        model.train()

        if self._optimizer is None:
            self._optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=1e-5
            )
        return None 

    def train_epoch(self, model, dataset):
        torch.cuda.empty_cache()

        # swap dataset
        dataset.swap()

        # init data loader
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4
        )   

        # training loop
        idx_to_cardinalities = model.idx_to_cardinalities

        for step, data in enumerate(data_loader):
            x, y = to_device(data, self.device)

            # forward
            _, dec = model(x)

            # calc losses 
            self._optimizer.zero_grad()
            loss = 0

            for idx in range(len(dec)):
                if idx_to_cardinalities is not None and idx in idx_to_cardinalities:
                    loss += torch.nn.CrossEntropyLoss()(dec[idx], y[:, idx].long())
                else:
                    loss += torch.nn.MSELoss()(dec[idx].squeeze(), y[:, idx].float()) / dec[idx].squeeze().std()
            
            show_message('[Step][{}][Loss][{}]'.format(step, loss.item()), 2, self.verbose)
            loss.backward()
            self._optimizer.step()    
            self._step_meter.update({'loss': loss.item()})
        
        return np.mean(self._step_meter['loss'])

    @property
    def epoch_history(self) -> Dict:
        return self._epoch_meter.history

    @property
    def step_history(self) -> Dict:
        return self._step_meter.history


def to_device(data: torch.Tensor, device: str) -> List[torch.Tensor]:
    r"""
    Moves the batch to the correct device. The returned batch is of the same type as the input batch, 
    just having all tensors on the correct device.
    
    Args:
        data (torch.Tensor or list of them):
            Input batch data.
        
        device (str):
            The target device.
    
    Returns:
        batch_data (torch.Tensor or list of them):
            Batch data on the correct device.
    
    """ 
    if isinstance(data, torch.Tensor):
        return data.to(device).float()

    if not all(isinstance(d, torch.Tensor) for d in data):
        raise TypeError('All elements in `data` must be `torch.Tensor`')

    return [d.to(device).float() for d in data]


__all__ = [
    'SwapDAEPretrainer'
]
