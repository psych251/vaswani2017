import torch
import torch.nn as nn
from torch.nn import ReLU, Tanh
import numpy as np
import time
import os
import torch.nn.functional as F
from crab.custom_modules import *

d = {i:"cuda:"+str(i) for i in range(torch.cuda.device_count())}
DEVICE_DICT = {-1:"cpu", **d}

class EncodingBlock(nn.Module):
    def __init__(self, emb_size, attn_size, n_heads=8,
                                            act_fxn="ReLU",
                                            prob_embs=False,
                                            prob_attn=False):
        """
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        act_fxn: str
            the name of the activation function to be used with the
            MLP
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        """
        super().__init__()
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_heads = n_heads
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn

        self.norm0 = nn.LayerNorm((emb_size,))
        self.multi_attn = MultiHeadAttention(emb_size=emb_size,
                                             attn_size=attn_size,
                                             n_heads=n_heads,
                                             prob_attn=self.prob_attn)
        self.norm1 = nn.LayerNorm((emb_size,))
        self.fwd_net = nn.Sequential(nn.Linear(emb_size,emb_size),
                                     globals()[act_fxn](),
                                     nn.Linear(emb_size,emb_size))
        self.norm2 = nn.LayerNorm((emb_size,))
        if self.prob_embs:
            self.proj = nn.Sequential(nn.Linear(emb_size,2*emb_size),
                                      globals()[act_fxn](),
                                      nn.Linear(2*emb_size,2*emb_size))

    def forward(self, x, mask=None, x_mask=None):
        """
        x: torch FloatTensor (B,S,E)
            batch by seq length by embedding size
        mask: torch float tensor (B,S,S) (optional)
            if a mask is argued, it is applied to the attention values
            to prevent information contamination
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        x = self.norm0(x)
        fx = self.multi_attn(q=x,k=x,v=x,mask=mask)
        fx = self.norm1(fx+x)
        fx = self.fwd_net(fx)
        fx = fx+self.norm2(fx+x)
        if self.prob_embs:
            fx = self.proj(fx)
        return fx

class Encoder(nn.Module):
    def __init__(self, seq_len, emb_size, attn_size, n_layers,n_heads,
                                                     use_mask=False,
                                                     act_fxn="ReLU",
                                                     prob_embs=False,
                                                     prob_attn=False):
        """
        seq_len: int
            the length of the sequences to be analyzed
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_layers: int
            the number of encoding layers
        n_heads: int
            the number of attention heads
        use_mask: bool
            if true, creates a mask to prevent the model from peaking
            ahead
        act_fxn: str
            the name of the activation function to be used with the
            MLP
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        """
        super().__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.act_fxn = act_fxn
        self.use_mask = use_mask
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn

        self.pos_encoding = PositionalEncoder(seq_len, emb_size)
        mask = self.get_mask(x.shape[1]) if self.use_mask else None
        self.register_buffer("mask",mask)
        self.enc_layers = nn.ModuleList([])
        for _ in range(n_layers):
            block = EncodingBlock(emb_size=emb_size,
                                  attn_size=attn_size,
                                  n_heads=n_heads,
                                  act_fxn=act_fxn,
                                  prob_embs=self.prob_embs,
                                  prob_attn=self.prob_attn)
            self.enc_layers.append(block)

    def forward(self,x,x_mask=None):
        """
        x: torch FloatTensor (B,S,E)
            batch by seq length by embedding size
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        if x.shape[1] != self.seq_len:
            mask = self.get_mask(x.shape[1])
            mask = mask.to(DEVICE_DICT[x.get_device()])
            self.register_buffer("mask",mask)
        else:
            mask = self.mask
        if self.prob_embs:
            x = sample_probs(x)
        fx = self.pos_encoding(x)
        for i,enc in enumerate(self.enc_layers):
            fx = enc(fx,mask=mask,x_mask=x_mask)
            if self.prob_embs and i < len(self.enc_layers)-1:
                fx = sample_probs(fx)
        return fx

    def get_mask(self, seq_len):
        mask = torch.FloatTensor(np.tri(seq_len,seq_len))
        mask = mask.masked_fill(mask==0,-1e10)
        mask[mask==1] = 0
        return mask

class DecodingBlock(nn.Module):
    def __init__(self, emb_size, attn_size, n_heads=8,
                                            act_fxn="ReLU",
                                            prob_embs=False,
                                            prob_attn=False):
        """
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        act_fxn: str
            the name of the activation function to be used with the
            MLP
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        """
        super().__init__()
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn

        self.norm0 = nn.LayerNorm((emb_size,))
        self.multi_attn1 = MultiHeadAttention(emb_size=emb_size,
                                              attn_size=attn_size,
                                              n_heads=n_heads,
                                              prob_attn=self.prob_attn)
        self.norm1 = nn.LayerNorm((emb_size,))

        self.multi_attn2 = MultiHeadAttention(emb_size=emb_size,
                                              attn_size=attn_size,
                                              n_heads=n_heads,
                                              prob_attn=self.prob_attn)
        self.norm2 = nn.LayerNorm((emb_size,))

        self.fwd_net = nn.Sequential(nn.Linear(emb_size, emb_size),
                                     globals()[act_fxn](),
                                     nn.Linear(emb_size,emb_size))
        self.norm3 = nn.LayerNorm((emb_size,))

        if self.prob_embs:
            self.proj = nn.Sequential(nn.Linear(emb_size,2*emb_size),
                                      globals()[act_fxn](),
                                      nn.Linear(2*emb_size,2*emb_size))

    def forward(self, x, encs, mask=None, x_mask=None,
                                          enc_mask=None):
        """
        x: torch FloatTensor (B,S,E)
            batch by seq length by embedding size
        encs: torch FloatTensor (B,S,E)
            batch by seq length by embedding size of encodings
        mask: torch float tensor (B,S,S) (optional)
            if a mask is argued, it is applied to the attention values
            to prevent information contamination
        x_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        enc_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        """

        fx = self.multi_attn1(q=x,k=x,v=x,mask=mask,
                                          q_mask=x_mask,
                                          k_mask=x_mask)
        fx = self.norm1(fx+x)
        fx = self.multi_attn2(q=fx,k=encs,v=encs, mask=None,
                                                  q_mask=x_mask,
                                                  k_mask=enc_mask)
        fx = self.norm2(fx+x)
        fx = self.fwd_net(fx)
        fx = self.norm3(fx+x)
        if self.prob_embs:
            fx = self.proj(fx)
        return fx

class Decoder(nn.Module):
    def __init__(self, seq_len, emb_size, attn_size, n_layers,
                                                     n_heads,
                                                     use_mask=False,
                                                     act_fxn="ReLU",
                                                     gen_decs=False,
                                                     init_decs=False,
                                                     prob_embs=False,
                                                     prob_attn=False):
        """
        seq_len: int
            the length of the sequences to be analyzed
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_layers: int
            the number of encoding layers
        n_heads: int
            the number of attention heads
        use_mask: bool
            if true, a no-peak mask is applied so that elements later
            in the decoding sequence are hidden from elements earlier
            in the decoding
        gen_decs: bool
            if true, decodings are generated individually and used
            as the inputs for later decodings. (stands for generate
            decodings). This ensures earlier attention values are
            completely unaffected by later inputs.
        init_decs: bool
            if true, an initialization decoding vector is learned as
            the initial input to the decoder.
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        """
        super().__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.act_fxn = act_fxn
        self.use_mask = use_mask
        self.gen_decs = gen_decs
        self.init_decs = init_decs
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn

        mask = self.get_mask(seq_len, bidirectional=False) if use_mask\
                                                            else None
        self.register_buffer("mask", mask)
        self.pos_encoding = PositionalEncoder(seq_len, emb_size)
        if self.init_decs: 
            if self.prob_embs: vec_size = 2*self.emb_size
            else: vec_size = self.emb_size
            self.inits = torch.randn(1,1,vec_size)
            divisor = float(np.sqrt(vec_size))
            self.inits = nn.Parameter(self.inits/divisor)
        self.dec_layers = nn.ModuleList([])
        for _ in range(n_layers):
            block = DecodingBlock(emb_size=emb_size,
                                  attn_size=attn_size,
                                  n_heads=n_heads,
                                  act_fxn=act_fxn,
                                  prob_attn=self.prob_attn,
                                  prob_embs=self.prob_embs)
            self.dec_layers.append(block)

    def forward(self, x, encs, x_mask=None, enc_mask=None):
        """
        x: torch tensor (B,S,E)
            the decoding embeddings
        encs: torch tensor (B,S,E)
            the output from the encoding stack
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        enc_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        if (not self.train and not self.init_decs) or self.gen_decs:
            return self.gen_dec_fwd(x,encs, x_mask=x_mask,
                                            enc_mask=enc_mask)
        else:
            if self.mask is not None and x.shape[1] != self.seq_len:
                mask = self.get_mask(x.shape[1])
            else:
                mask = self.mask
            if self.init_decs:
                x = self.inits.repeat(len(x),x.shape[1],1)
                x_mask = None
            if self.prob_embs:
                x = sample_probs(x)
                og_encs = encs
                encs = sample_probs(og_encs)
            fx = self.pos_encoding(x)
            for i,dec in enumerate(self.dec_layers):
                fx = dec(fx, encs, mask=self.mask, x_mask=x_mask,
                                                   enc_mask=enc_mask)
                if self.prob_embs:
                    fx = sample_probs(fx)
                    encs = sample_probs(og_encs)
            return fx

    def gen_dec_fwd(self, x, encs, x_mask=None, enc_mask=None):
        """
        x: torch tensor (B,S,E)
            the decoding embeddings
        encs: torch tensor (B,S,E)
            the output from the encoding stack
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        enc_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        outputs = x[:,:1]
        if self.prob_embs:
            outputs = sample_probs(outputs)
            og_encs = encs
            encs = sample_probs(og_encs)
        for i in range(x.shape[1]-1):
            fx = outputs
            fx = self.pos_encoding(fx)
            for dec in self.dec_layers:
                fx = dec(fx, encs, mask=None, x_mask=x_mask,
                                              enc_mask=enc_mask)
            outputs = torch.cat([outputs,fx[:,-1:]],dim=1)
            if self.prob_embs:
                outputs = sample_probs(outputs)
                encs = sample_probs(og_encs)
        return outputs

    def get_mask(self, seq_len, bidirectional=False):
        """
        Returns a diagonal mask to prevent the transformer from looking
        at the word it is trying to predict.

        seq_len: int
            the length of the sequence
        bidirectional: bool
            if true, then the decodings will see everything but the
            current token.
        """
        if bidirectional:
            mask = torch.zeros(seq_len,seq_len)
            mask[range(seq_len-1),range(1,seq_len)] = -1e10
        else:
            mask = torch.FloatTensor(np.tri(seq_len,seq_len))
            mask = mask.masked_fill(mask==0,-1e10)
            mask[mask==1] = 0
        return mask


class TransformerBase(nn.Module):
    def __init__(self, seq_len=None, n_vocab=None,
                                     emb_size=512,
                                     enc_slen=None,
                                     dec_slen=None,
                                     attn_size=64,
                                     n_heads=8,
                                     enc_layers=6,
                                     dec_layers=6,
                                     enc_mask=False,
                                     class_h_size=4000,
                                     class_bnorm=True,
                                     class_drop_p=0,
                                     act_fxn="ReLU",
                                     enc_drop_p=0,
                                     dec_drop_p=0,
                                     ordered_preds=False,
                                     gen_decs=False,
                                     init_decs=False,
                                     idx_inputs=True,
                                     prob_embs=False,
                                     prob_attn=False,
                                     **kwargs):
        """
        seq_len: int or None
            the maximum length of the sequences to be analyzed. If None,
            dec_slen and enc_slen must not be None
        enc_slen: int or None
            the length of the sequences to be encoded
        dec_slen: int or None
            the length of the sequences to be decoded
        n_vocab: int
            the number of words in the vocabulary
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        enc_layers: int
            the number of encoding layers
        dec_layers: int
            the number of decoding layers
        enc_mask: bool
            if true, encoder uses a mask
        class_h_size: int
            the size of the hidden layers in the classifier
        class_bnorm: bool
            if true, the classifier uses batchnorm
        class_drop_p: float
            the dropout probability for the classifier
        act_fxn: str
            the activation function to be used in the MLPs
        enc_drop_ps: float or list of floats
            the dropout probability for each encoding layer
        dec_drop_ps: float or list of floats
            the dropout probability for each decoding layer
        ordered_preds: bool
            if true, the decoder will mask the predicted sequence so
            that the attention modules will not see the tokens ahead
            located further along in the sequence.
        gen_decs: bool
            if true, decodings are generated individually and used
            as the inputs for later decodings. (stands for generate
            decodings). This ensures earlier attention values are
            completely unaffected by later inputs.
        init_decs: bool
            if true, an initialization decoding vector is learned as
            the initial input to the decoder.
        idx_inputs: bool
            if true, the inputs are integer (long) indexes that require
            an embedding layer. Otherwise it is assumed that the inputs
            are feature vectors that do not require an embedding layer
        prob_embs: bool
            if true, all embedding vectors are treated as parameter
            vectors for gaussian distributions before being fed into
            the transformer architecture
        prob_attn: bool
            if true, the queries and keys are projected into a mu and
            sigma vector and sampled from a gaussian distribution
            before the attn mechanism
        """
        super().__init__()

        self.seq_len = seq_len
        self.enc_slen = enc_slen if enc_slen is not None else seq_len
        self.dec_slen = dec_slen if dec_slen is not None else seq_len
        self.n_vocab = n_vocab
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_heads = n_heads
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_mask = enc_mask
        self.class_bnorm = class_bnorm
        self.class_drop_p = class_drop_p
        self.class_h_size = class_h_size
        self.act_fxn = act_fxn
        self.enc_drop_p = enc_drop_p
        self.dec_drop_p = dec_drop_p
        self.ordered_preds = ordered_preds
        self.gen_decs = gen_decs
        self.init_decs = init_decs
        self.idx_inputs = idx_inputs
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn
    
    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

class Transformer(TransformerBase):
    def __init__(self, *args, **kwargs):
        """
        See TransformerBase for arguments
        """
        super().__init__(*args, **kwargs)
        self.transformer_type = SEQ2SEQ

        print("enc_slen:", self.enc_slen)
        print("dec_slen:", self.dec_slen)
        if self.prob_embs: 
            self.embeddings =nn.Embedding(self.n_vocab,2*self.emb_size)
        else:
            self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        self.encoder = Encoder(self.enc_slen,emb_size=self.emb_size,
                                            attn_size=self.attn_size,
                                            n_layers=self.enc_layers,
                                            n_heads=self.n_heads,
                                            use_mask=self.enc_mask,
                                            act_fxn=self.act_fxn,
                                            prob_attn=self.prob_attn,
                                            prob_embs=self.prob_embs)

        use_mask = not self.init_decs and self.ordered_preds
        self.decoder = Decoder(self.dec_slen,self.emb_size,
                                            self.attn_size,
                                            self.dec_layers,
                                            n_heads=self.n_heads,
                                            act_fxn=self.act_fxn,
                                            use_mask=use_mask,
                                            init_decs=self.init_decs,
                                            gen_decs=self.gen_decs,
                                            prob_attn=self.prob_attn,
                                            prob_embs=self.prob_embs)

        self.classifier = Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)
        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def forward(self, x, y):
        """
        x: float tensor (B,S)
        y: float tensor (B,S)
        """
        y_mask = (y==0).masked_fill(y==0,1e-10)
        self.embeddings.weight.data[0,:] = 0 # Mask index
        bitmask = (x==0)
        if self.idx_inputs is not None and self.idx_inputs:
            embs = self.embeddings(x)
            x_mask = bitmask.masked_fill(bitmask,1e-10)
        else:
            embs = x
            x_mask=None
        encs = self.encoder(embs,x_mask=x_mask)
        encs = self.enc_dropout(encs)
        dembs = self.embeddings(y)
        decs = self.decoder(dembs, encs, x_mask=y_mask,
                                         enc_mask=x_mask)
        decs = self.dec_dropout(decs)
        decs = decs.reshape(-1,decs.shape[-1])
        preds = self.classifier(decs)
        return preds.reshape(len(y),y.shape[1],preds.shape[-1])

class Classifier(nn.Module):
    def __init__(self, emb_size, n_vocab, h_size, bnorm=True,
                                                  drop_p=0,
                                                  act_fxn="ReLU"):
        """
        emb_size: int
            the size of the embedding layer
        n_vocab: int
            the number of words in the vocabulary
        h_size: int
            the size of the hidden layer
        bnorm: bool
            if true, the hidden layers use a batchnorm layer
        drop_p: float
            the dropout probability of the dropout modules
        act_fxn: str
            the name of the activation function to be used with the
            MLP
        """

        super().__init__()
        self.emb_size = emb_size
        self.n_vocab = n_vocab
        self.h_size = h_size
        self.bnorm = bnorm
        self.drop_p = drop_p
        self.act_fxn = act_fxn

        modules = []
        modules.append(nn.Linear(emb_size,h_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.Dropout(drop_p))
        modules.append(globals()[act_fxn]())

        modules.append(nn.Linear(h_size,h_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.Dropout(drop_p))
        modules.append(globals()[act_fxn]())

        modules.append(nn.Linear(h_size,n_vocab))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

