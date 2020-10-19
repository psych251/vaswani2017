import numpy as np
import torch
import json
import os
import cv2

def get_max_key(d):
    """
    Returns key corresponding to maxium value

    d: dict
        keys: object
        vals: int or float
    """
    max_v = -np.inf
    max_k = None
    for k,v in d.items():
        if v > max_v:
            max_v = v
            max_k = k
    return max_k

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def sample_probs(mu_sig, min_sig=0.00001, dim=-1):
    """
    samples from a gaussian distribution using the parameters
    stored in the latns

    mu_sig: torch FloatTensor (...,Mu+Sig)
        the means and the standard deviations for a gaussian.
        the vector will be split halfway on the argued dimension
        with the assumption that the means are located first.
    min_sig: float
        the minimum value sigma can take on
    dim: int
        the dimension that the split should occur on
    """
    mu,sig = torch.chunk(mu_sig, 2, dim=dim)
    sig = F.softplus(sig)+min_sig
    return mu + torch.randn_like(sig)*sig

