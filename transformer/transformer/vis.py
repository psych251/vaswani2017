import torch
import numpy as np
import time
import os
import torch.nn.functional as F
from crab.custom_modules import MultiHeadAttention

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def get_attn_mtxs(model, x=None, y=None, to_numpy=False):
    """
    returns a lit of all attention matrices for each multi-head attn in
    the model

    model: torch Module
    x: torch tensor or None
        FloatTensor or LongTensor depending on model type. if None, ps
        from model attentions are collected
    y: torch tensor
        FloatTensor or LongTensor or None depending on model
    to_numpy: bool
        if true, all tensors are returned as ndarrays
    """
    if x is not None:
        _ = model(x,y)
    attn_mtxs = dict()
    for name,modu in model.named_modules():
        if isinstance(modu,MultiHeadAttention):
            attn_mtxs[name] = modu.ps.detach().cpu()
            if to_numpy: attn_mtxs[name] = attn_mtxs[name].numpy()
    return attn_mtxs

"""
for each attn matrix:
    average across heads?
    matrix multiply single row with prev attention matrix
    average to one row
    repeat
"""

def get_attn_map(attn_mtxs, start_keys, idx, attn_keys, to_numpy=False):
    """
    returns an attention map for the first multiheaded attention in the
    model.

    attn_mtxs: dict of tensors or ndarrays
        keys: str
            name of layer that produced this particular attention
        vals: ndarray or tensor (H,N,M)
            the attention matrix with number of heads as the first dim
    start_keys: list of strs
        the names of the first attention matrices to focus on. Will
        average over each of these attn matrices
    idx: int
        the index of the slot to visualize attention for 
    attn_keys: list of str
        the keys of all of the attn layers leading up to the layer and
        index of interest. Important that these keys are ordered from
        the start key backwards towards the earlier parts of the model.
    to_numpy: bool
        if true, all tensors are returned as ndarrays
    """
    attn = attn_mtxs[start_keys[0]].mean(0)[idx:idx+1]
    for i in range(1,len(start_keys)):
        attn += attn_mtxs[start_keys[i]].mean(0)[idx:idx+1]
    attn = (attn/len(start_keys)).T # (M,1)
    for key in attn_keys:
        mtx = attn_mtxs[key].mean(0) # (M,M)
        matmul = mtx.T@attn # (M,1)
        attn = matmul/mtx.shape[0]
    if to_numpy and isinstance(attn, torch.tensor): 
        return attn.data.cpu().numpy()
    return attn

