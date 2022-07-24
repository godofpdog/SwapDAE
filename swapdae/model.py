import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Union, List, Dict


class Mish(nn.Module):
    def __init__(self) -> None:
        super(Mish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x * nn.functional.sigmoid(x)


def _get_activation(activation: str = None) -> nn.Module:
    if activation is None:
        return nn.Identity()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'mish':
        return Mish()
    elif activation == 'swish':
        return Swish()
    else:
        raise ValueError('Not supported activation')


class LinearBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout_rate: float, 
        activation: str = None,
        is_bias: bool = False,
    ) -> None:
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=is_bias)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = _get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        activation: str = None,
        is_bias: bool = False
    ) -> None:
        super(ResBlock, self).__init__()
        self.linear1 = LinearBlock(input_dim, hidden_dim, dropout_rate, activation, is_bias)
        self.linear2 = LinearBlock(hidden_dim, output_dim, dropout_rate, activation, is_bias)
        self.proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        s = self.proj(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x + s


class EmbeddingEncoder(nn.Module):
    def __init__(
        self, 
        input_dims: int, 
        cate_indices: int, 
        cate_dims: int, 
        embed_dims: int
    ) -> None:
        super(EmbeddingEncoder, self).__init__()
        self._is_skip = False 

        if cate_indices is None:
            cate_indices = []

        if cate_dims is None:
            cate_dims = []

        if embed_dims is None:
            embed_dims = 1
        
        if isinstance(cate_indices, int):
            cate_indices = [cate_indices]

        if isinstance(cate_dims, int):
            cate_dims = [cate_dims]

        if cate_indices == [] or cate_dims == []:
            self._is_skip = True 
            self.output_dims = input_dims
            return 

        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(cate_indices)

        if len(cate_indices) != len(embed_dims):
            raise ValueError('`cate_indices` and `embed_dims` must have same length, but got {} and {}.'\
                .format(len(cate_indices), len(embed_dims)))
        
        self.sorted_indices = np.argsort(cate_indices)
        self.cate_indices = [cate_indices[i] for i in self.sorted_indices]
        self.cate_dims = [cate_dims[i] for i in self.sorted_indices]
        self.embed_dims = [embed_dims[i] for i in self.sorted_indices]
        self.output_dims = int(input_dims + np.sum(embed_dims) - len(embed_dims))

        # build models
        self.embedding_layers = nn.ModuleList()
    
        for cate_dim, embed_dim in zip(self.cate_dims, self.embed_dims):
            self.embedding_layers.append(
                nn.Embedding(cate_dim, embed_dim)
            )

        # conti indices
        self.conti_indices = torch.ones(input_dims, dtype=torch.bool)
        self.conti_indices[self.cate_indices] = 0

    def forward(self, x):
        outputs = []
        cnt = 0

        if self._is_skip:
            return x

        for i, is_conti in enumerate(self.conti_indices):
            if is_conti:
                outputs.append(
                    x[:, i].float().view(-1, 1)
                )
            else:
                outputs.append(
                    self.embedding_layers[cnt](x[:, i].long())
                )
                cnt +=1
        return torch.cat(outputs, dim=1)


class DecoderHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        idx_to_cardinalities: Dict
    ) -> None:
        super(DecoderHead, self).__init__()
        self.heads = nn.ModuleList()

        for i in range(input_dim):
            if idx_to_cardinalities is not None and i in idx_to_cardinalities:
                head = nn.Linear(hidden_dim, idx_to_cardinalities[i], bias=False)
            else:
                head = nn.Linear(hidden_dim, 1, bias=False)
        
            self.heads.append(head)
        
    def forward(self, x: Tensor) -> Tensor:
        outputs = []

        for head in self.heads:
            outputs.append(head(x))
        return outputs


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: Union[int, List[int]],
        decoder_hidden_dims: Union[int, List[int]],
        cate_indices: List[int] = None, 
        cardinalities: List[int] = None,
        cate_embedding_dim: int = 1,
        dropout_rate: float = 0.0,
        activation: str = 'mish',
        is_bias: bool = False,
        block_type: str = 'mlp',
        is_reprs_norm: bool = False
    ) -> None:
        super(AutoEncoder, self).__init__()
        assert block_type in ('mlp', 'res'), 'Not supported block type.'
        assert cate_embedding_dim > 0, '``ccate_embedding_dim`` must > 0.'

        if cate_indices is not None and cardinalities is None:
            raise ValueError('Must define ``cardinalities`` when ``cate_indices`` is not None.')
        elif cardinalities is not None and cate_indices is None:
            raise ValueError('Must define ``cate_indices`` when ``cardinalities`` is not None.')
        elif cate_indices is not None and cardinalities is not None:
            if len(cate_indices) != len(cardinalities):
                raise ValueError('Length of ``cate_indices`` must be equal to the length of ``cardinalities``.')

        self.embedding = EmbeddingEncoder(input_dim, cate_indices, cardinalities, cate_embedding_dim)

        if cate_indices is not None:
            enc_input_dim = input_dim + len(cardinalities) * (cate_embedding_dim - 1)
            dec_input_dim = encoder_hidden_dims[-1]

        self.encoder = self._build_layers(enc_input_dim, encoder_hidden_dims, dropout_rate, activation, is_bias, block_type)
        self.decoder = self._build_layers(encoder_hidden_dims[-1], decoder_hidden_dims, dropout_rate, activation, is_bias, block_type)

        self.idx_to_cardinalities = {k: v for k, v in zip(cate_indices, cardinalities)} if cate_indices is not None else None
        self.decoder_head = DecoderHead(input_dim, decoder_hidden_dims[-1], self.idx_to_cardinalities)

        self.is_reprs_norm = is_reprs_norm

    def _build_layers(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        dropout_rate: float,
        activation: str,
        is_bias: bool,
        block_type: str
    ) -> nn.Sequential:
        layers = []
        _input_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(
                LinearBlock(
                    _input_dim, h_dim, dropout_rate, activation, is_bias
                ) if block_type == 'mlp' else ResBlock(
                    _input_dim, h_dim, hidden_dim, dropout_rate, activation, is_bias
                ) 
            )
            _input_dim = h_dim
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        enc = self.encoder(emb)

        if self.is_reprs_norm:
            enc = nn.functional.normalize(enc, p=2, dim=-1)

        dec = self.decoder(enc)
        dec = self.decoder_head(dec)
        
        return enc, dec


__all__ = [
    'AutoEncoder'
]
