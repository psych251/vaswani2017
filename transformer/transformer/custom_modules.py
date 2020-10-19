import torch
import torch.nn as nn
from torch.nn import ReLU, Tanh, Softplus
import torch.nn.functional as F
import numpy as np
from transformer.utils import sample_probs

d = {i:"cuda:"+str(i) for i in range(torch.cuda.device_count())}
DEVICE_DICT = {-1:"cpu", **d}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, emb_size):
        """
        Creates an additive to ensure the positional information of
        the words is known to the transformer

        seq_len: int
            the length of the sequence
        emb_size: int
            the dimensionality of the embeddings
        """
        super().__init__()
        pos_enc = self.get_pos_encoding(seq_len,emb_size)
        self.register_buffer('pos_enc',pos_enc)
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.pos_enc.requires_grad = False

    def forward(self, x):
        try:
            return self.pos_enc[:x.shape[1]] + x
        except:
            print("Increasing length of positional encoding")
            pos_enc = self.get_pos_encoding(x.shape[1],self.emb_size)
            pos_enc = pos_enc.to(DEVICE_DICT[x.get_device()])
            fx = pos_enc[:x.shape[1]] + x
            self.register_buffer('pos_enc',pos_enc)
            return fx

    def get_pos_encoding(self, seq_len, emb_size):
        """
        seq_len: int
            the length of the sequences to be analyzed
        emb_size: int
            the size of the embeddings

        Returns:
            pos_enc: tensor (seq_len, emb_size)
                the positional encodings to be summed with the input
                matrix
        """
        pos_enc = torch.arange(seq_len)[:,None].expand(seq_len,emb_size)
        scaling = torch.arange(emb_size//2)[None].repeat(2,1)
        scaling = scaling.transpose(1,0).reshape(-1).float()
        scaling = 10000**(scaling/emb_size)
        pos_enc = pos_enc.float()/scaling
        pos_enc[:,::2] = torch.sin(pos_enc[:,::2])
        pos_enc[:,1::2] = torch.cos(pos_enc[:,1::2])
        return pos_enc

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, attn_size, n_heads=8,
                                            prob_attn=False):
        """
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        """
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.prob_attn = prob_attn
        self.attn_size = attn_size
        self.qk_attn_size = 2*attn_size if prob_attn else attn_size

        xavier_scale = np.sqrt((emb_size + attn_size*n_heads)/2)
        w_v = torch.randn(emb_size, attn_size*n_heads)/xavier_scale
        self.w_v = nn.Parameter(w_v)

        attnhead = self.qk_attn_size*n_heads
        w_q = torch.randn(emb_size, attnhead)/xavier_scale
        self.w_q = nn.Parameter(w_q)
        w_k = torch.randn(emb_size, attnhead)/xavier_scale
        self.w_k = nn.Parameter(w_k)

        self.outs = nn.Linear(attn_size*n_heads, emb_size)
        self.ps = None

    def forward(self, q, k, v, mask=None, q_mask=None,
                                          k_mask=None):
        """
        q: tensor (B,S,E)
            the queries
        k: tensor (B,S,E)
            the keys
        v: tensor (B,S,E)
            the values
        mask: tensor (S,S)
            a mask consisting of 0s and some large magnitude negative
            number in the positions that should not influence the
            attention.
            [0,-1e9,-1e9,-1e9]
            [0,   0,-1e9,-1e9]
            [0,   0,   0,-1e9]
            [0,   0,   0,   0]
        q_mask: tensor (B,S)
            a mask consisting of 0s and some large magnitude negative
            number in the positions that should not influence the
            attention for each query.
            [0,-1e9,-1e9,-1e9]
        k_mask: tensor (B,S)
            a mask consisting of 0s and some large magnitude negative
            number in the positions that should not influence the
            attention for each query.
            [0,-1e9,-1e9,-1e9]
        """
        fq = torch.matmul(q, self.w_q) # (B,S,H*A)
        fk = torch.matmul(k, self.w_k) # (B,S,H*A)
        fv = torch.matmul(v, self.w_v) # (B,S,H*A)

        batch,seq_q = q.shape[:2]
        fq = fq.reshape(batch, seq_q, self.n_heads, self.qk_attn_size) 
        fq = fq.permute(0,2,1,3) # (B,H,Sq, A)

        batch,seq_k = k.shape[:2]
        fk = fk.reshape(batch, seq_k, self.n_heads, self.qk_attn_size) 
        fk = fk.permute(0,2,3,1) # (B,H,A,Sk)
        # seq_k should always be equal to seq_v
        fv = fv.reshape(batch, seq_k, self.n_heads, self.attn_size) 
        fv = fv.permute(0,2,1,3) # (B,H,Sk,A)

        if self.prob_attn:
            fk = sample_probs(fk,dim=-2)
            fq = sample_probs(fq)

        f = torch.matmul(fq,fk)/np.sqrt(self.attn_size) # (B,H,Sq,Sk)
        if mask is not None:
            row,col = f.shape[-2:]
            f = f + mask[:row,:col]
        new_min = torch.min(f).item()
        if q_mask is not None:
            f = f + q_mask[:,None,:f.shape[-2],None]
        if k_mask is not None:
            f = f + k_mask[:,None,None,:f.shape[-1]]
        f = torch.clamp(f,new_min,torch.max(f).item())
        ps = F.softmax(f,dim=-1) # (B,H,Sq,Sk) ps along Sk
        self.ps = ps
        attns = torch.matmul(ps,fv) # (B,H,Sq,A)
        attns = attns.permute(0,2,1,3) # (B,Sq,H,A)
        attns = attns.reshape(batch,seq_q,-1)
        return self.outs(attns)


class Sin(nn.Module):
    def __init__(self):
        """
        A sinusoidal activation function.
        """
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class Multiply(nn.Module):
    """
    This is a wrapper module to multiply the activations be a specific
    amount.
    """
    def __init__(self, multiplier):
        """
        multiplier: float or FloatTensor
            the value to multiply the activations with
        """
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x*self.multiplier
