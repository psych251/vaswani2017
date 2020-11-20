import torch
import torch.nn as nn
from torch.nn import ReLU, Tanh
import numpy as np
import time
import os
import torch.nn.functional as F
from transformer.custom_modules import *

d = {i:"cuda:"+str(i) for i in range(torch.cuda.device_count())}
DEVICE_DICT = {-1:"cpu", **d}

class ECoderBase(nn.Module):
    """
    This is a base class to consolidate most of the common parameters
    in the encoder and decoder modules. This makes it easier to make
    variations for other projects.
    """
    def __init__(self, seq_len, emb_size, attn_size, n_layers,
                                                     n_heads,
                                                     use_mask=False,
                                                     act_fxn="ReLU",
                                                     prob_embs=False,
                                                     prob_attn=False,
                                                     drop_p=0,
                                                     **kwargs):
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
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        drop_p: float
            the dropout probability
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
        self.drop_p = drop_p

class Encoder(ECoderBase):
    def __init__(self, *args, **kwargs):
        """
        See ECoderBase for argument details

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
        drop_p: float
            the dropout probability
        """
        super().__init__(*args, **kwargs)
        if self.use_mask:
            print("encoder is using a mask, this is likely undesired!")

        self.dropout = nn.Dropout(self.drop_p)
        self.pos_encoding = PositionalEncoder(self.seq_len,self.emb_size)
        mask = self.get_mask(self.seq_len) if self.use_mask else None
        self.register_buffer("mask",mask)
        self.enc_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            block = EncodingBlock(emb_size=self.emb_size,
                                  attn_size=self.attn_size,
                                  n_heads=self.n_heads,
                                  act_fxn=self.act_fxn,
                                  prob_embs=self.prob_embs,
                                  prob_attn=self.prob_attn,
                                  drop_p=self.drop_p)
            self.enc_layers.append(block)


    def forward(self,x,x_mask=None):
        """
        x: torch FloatTensor (B,S,E)
            batch by seq length by embedding size
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        # Resize mask in case x is of unusual sequence length
        if self.mask is not None and x.shape[1] != self.mask.shape[0]:
            mask = self.get_mask(x.shape[1])
            mask = mask.cuda(non_blocking=True)
            self.register_buffer("mask",mask)
        else:
            mask = self.mask
        # Sample embedding vectors if using stochastic paradigm
        if self.prob_embs:
            x = sample_probs(x)

        # Encode
        fx = self.pos_encoding(x)
        fx = self.dropout(fx)
        for i,enc in enumerate(self.enc_layers):
            fx = enc(fx,mask=mask,x_mask=x_mask)
            if self.prob_embs and i < len(self.enc_layers)-1:
                fx = sample_probs(fx)
        return fx

    def get_mask(self, seq_len):
        """
        Creates mask that looks like the following with rows and columns
        of length seq_len:

        [0,-1e9,-1e9,-1e9]
        [0,   0,-1e9,-1e9]
        [0,   0,   0,-1e9]
        [0,   0,   0,   0]

        seq_len: int
            the side length of the mask square
        """
        mask = torch.FloatTensor(np.tri(seq_len,seq_len))
        mask = mask.masked_fill(mask==0,-1e10)
        mask[mask==1] = 0
        return mask

class Decoder(ECoderBase):
    def __init__(self, *args, init_decs=False, multi_init=False,
                                               **kwargs):
        """
        See ECoderBase for details on kwarg arguments

        init_decs: bool
            if true, an initialization decoding vector is learned as
            the initial input to the decoder.
        multi_init: bool
            if true, the initialization vector has a unique value for
            each slot in the generated sequence. Only applies if
            init_decs is true
        """
        super().__init__(*args, **kwargs)
        self.init_decs = init_decs
        self.multi_init = multi_init

        mask = self.get_mask(self.seq_len, bidirectional=False) if\
                                                self.use_mask else None
        self.register_buffer("mask", mask)

        self.dropout = nn.Dropout(self.drop_p)
        self.pos_encoding = PositionalEncoder(self.seq_len,
                                              self.emb_size)
        # init_decs is a paradigm in which the decoded vectors all
        # start from a learned initialization vector. multi_init is
        # an additional argument that allows each decoding location
        # in the sequence to use its own unique initialization
        if self.init_decs: 
            if self.prob_embs: vec_size = 2*self.emb_size
            else: vec_size = self.emb_size
            n_vecs = self.seq_len if self.multi_init else 1
            self.inits = torch.randn(1,n_vecs,vec_size)
            divisor = float(np.sqrt(vec_size))
            self.inits = nn.Parameter(self.inits/divisor)
        self.dec_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            block = DecodingBlock(emb_size=self.emb_size,
                                  attn_size=self.attn_size,
                                  n_heads=self.n_heads,
                                  act_fxn=self.act_fxn,
                                  prob_attn=self.prob_attn,
                                  prob_embs=self.prob_embs,
                                  drop_p=self.drop_p)
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
        # Learned initialization paradigm
        if self.init_decs:
            x_len = self.seq_len if x is None else x.shape[1]
            n_vecs = 1 if self.multi_init else x_len
            x = self.inits.repeat(len(encs),n_vecs,1)
            x_mask = None

        if self.mask is not None and x.shape[1] != self.seq_len:
            mask = self.get_mask(x.shape[1])
        else:
            mask = self.mask

        if self.prob_embs:
            x = sample_probs(x)
            og_encs = encs
            encs = sample_probs(og_encs)

        fx = self.pos_encoding(x)
        fx = self.dropout(fx)
        for i,dec in enumerate(self.dec_layers):
            fx = dec(fx, encs, mask=self.mask, x_mask=x_mask,
                                               enc_mask=enc_mask)
            if self.prob_embs:
                fx = sample_probs(fx)
                encs = sample_probs(og_encs)
        return fx

    def get_mask(self, seq_len, bidirectional=False):
        """
        Returns a mask to prevent the transformer from using forbidden
        information >:)

        The mask looks like the following with rows and columns of
        length seq_len:

        [0,-1e9,-1e9,-1e9]
        [0,   0,-1e9,-1e9]
        [0,   0,   0,-1e9]
        [0,   0,   0,   0]

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

class Gencoder(ECoderBase):
    def __init__(self, *args, multi_init=False, **kwargs):
        """
        A decoder that uses learned vectors as decodeable outputs
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
        multi_init: bool
            if true, the initialization vector has a unique value for
            each slot in the generated sequence
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        drop_p: float
            dropout probability
        """
        super().__init__(*args, **kwargs)
        self.multi_init = multi_init
        self.drop_p = drop_p

        self.dropout = nn.Dropout(self.drop_p)

        if self.prob_embs: vec_size = 2*self.emb_size
        else: vec_size = self.emb_size
        if multi_init:
            self.init_vec = torch.randn(1,self.seq_len,vec_size)
        else:
            self.init_vec = torch.randn(1,1,vec_size)
        divisor = float(np.sqrt(vec_size))
        self.init_vec = nn.Parameter(self.init_vec/divisor)
        self.pos_encoding = PositionalEncoder(self.seq_len, 
                                              self.emb_size)
        self.dec_layers = nn.ModuleList([])
        block = GencodingBlock(emb_size=self.emb_size,
                              attn_size=self.attn_size,
                              n_heads=self.n_heads,
                              act_fxn=self.act_fxn,
                              prob_embs=self.prob_embs,
                              prob_attn=self.prob_attn,
                              drop_p=self.drop_p)
        self.dec_layers.append(block)
        for _ in range(self.n_layers-1):
            block = DecodingBlock(emb_size=self.emb_size,
                                  attn_size=self.attn_size,
                                  n_heads=self.n_heads,
                                  act_fxn=self.act_fxn,
                                  prob_embs=self.prob_embs,
                                  prob_attn=self.prob_attn,
                                  drop_p=self.drop_p)
            self.dec_layers.append(block)

    def get_init_vec(self, batch_size):
        """
        batch_size: int
            the size of the batch
        """
        if self.multi_init:
            return self.init_vec.repeat(batch_size,1,1)
        return self.init_vec.repeat(batch_size,self.seq_len,1)

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        x: torch tensor (B,S,E)
            the decoding embeddings
        y: torch tensor (B,S,E)
            the output from the encoding stack. if prob_embs is true,
            y is sampled within the forward fxn
        x_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        y_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        if x is None:
            x = self.get_init_vec(len(y))
        if self.prob_embs:
            x = sample_probs(x)
            og_y = y
            y = sample_probs(og_y)
        fx = self.pos_encoding(x)
        fx = self.dropout(fx)
        for dec in self.dec_layers:
            fx = dec(fx,y,x_mask=x_mask,enc_mask=y_mask)
            if self.prob_embs:
                fx = sample_probs(fx)
                y = sample_probs(og_y)
        return fx

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

class Attncoder(ECoderBase):
    """
    Very similar to the Decoder and Gencoder modules. This one is
    built for cases in which you do not want to apply self attention
    over the elements of interest but only attention over some other
    sequential vector set.
    """
    def __init__(self, *args, init_decs=False, multi_init=False,
                                               **kwargs):
        """
        A decoder that uses learned vectors as decodeable outputs
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
        init_decs: bool
            determines if you would like to have a learned initialization
            vector for the query vector.
        multi_init: bool
            if true, the initialization vector has a unique value for
            each slot in the generated sequence
        prob_embs: bool
            if true, embedding vectors are treated as parameter
            vectors for gaussian distributions
        prob_attn: bool
            if true, the queries and keys are projected into a
            gaussian parameter vectors space and sampled
        drop_p: float
            dropout probability
        """
        super().__init__(*args, **kwargs)
        self.init_decs = init_decs
        self.multi_init = multi_init
        print("probably want to add positional encoder to attncoder")

        if self.prob_embs: vec_size = 2*self.emb_size
        else: vec_size = self.emb_size
        if self.init_decs:
            if multi_init:
                self.init_vec = torch.randn(1,self.seq_len,vec_size)
            else:
                self.init_vec = torch.randn(1,1,vec_size)
            divisor = float(np.sqrt(vec_size))
            self.init_vec = nn.Parameter(self.init_vec/divisor)

        self.attn_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            block = AttnBlock(emb_size=self.emb_size,
                                  attn_size=self.attn_size,
                                  n_heads=self.n_heads,
                                  act_fxn=self.act_fxn,
                                  prob_embs=self.prob_embs,
                                  prob_attn=self.prob_attn,
                                  drop_p=self.drop_p)
            self.attn_layers.append(block)

    def get_init_vec(self, batch_size):
        """
        batch_size: int
            the size of the batch
        """
        if self.multi_init:
            return self.init_vec.repeat(batch_size,1,1)
        return self.init_vec.repeat(batch_size,self.seq_len,1)

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        x: torch tensor (B,S,E)
            the decoding embeddings
        y: torch tensor (B,S,E)
            the output from the encoding stack. if prob_embs is true,
            y is sampled within the forward fxn
        x_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        y_mask: torch FloatTensor (B,S)
            a mask to prevent some tokens from influencing the output
        """

        # Learned initialization paradigm
        if x is None or self.init_decs:
            x = self.get_init_vec(len(y))
        # Stochastic paradigm
        if self.prob_embs:
            x = sample_probs(x)
            og_y = y
            y = sample_probs(og_y)
        # Multiple layers of attention over the sequential vectors.
        # No self attention
        for block in self.attn_layers:
            x = block(x,y,x_mask=x_mask,enc_mask=y_mask)
            if self.prob_embs:
                x = sample_probs(x)
                y = sample_probs(og_y)
        return x

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

class AppendEncoder(ECoderBase):
    """
    This is an encoder that concatenates initialized vectors to the
    sequence to allow for encoding and decoding at the same time. It
    is effectively a Decoder only transformer but some fixed number
    of vectors are appended to the initial inputs for each run-through.
    """
    def __init__(self, *args, state_size=3, **kwargs):
        """
        See ECoderBase for argument details

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
        state_size: int
            the number of new vectors to append to the inputs for
            encoding.
        multi_init: bool
            if true, the initialization vector has a unique value for
            each slot in the generated sequence
        drop_p: float
            dropout probability
        """
        super().__init__(*args, **kwargs)
        self.state_size = state_size
        self.multi_init = multi_init

        self.dropout = nn.Dropout(self.drop_p)

        if self.use_mask:
            print("AppendEncoder is using a mask!!")

        if self.prob_embs: vec_size = 2*self.emb_size
        else: vec_size = self.emb_size
        if multi_init:
            self.init_vec = torch.randn(1,self.state_size,vec_size)
        else:
            self.init_vec = torch.randn(1,1,vec_size)
        divisor = float(np.sqrt(vec_size))
        self.init_vec = nn.Parameter(self.init_vec/divisor)

        self.n_encs = self.seq_len+self.state_size
        self.pos_encoding = PositionalEncoder(self.n_encs,self.emb_size)
        mask = self.get_mask(self.seq_len) if self.use_mask else None
        self.register_buffer("mask",mask)
        self.enc_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            block = EncodingBlock(emb_size=self.emb_size,
                                  attn_size=self.attn_size,
                                  n_heads=self.n_heads,
                                  act_fxn=self.act_fxn,
                                  prob_embs=self.prob_embs,
                                  prob_attn=self.prob_attn,
                                  drop_p=self.drop_p)
            self.enc_layers.append(block)

    def forward(self,x,x_mask=None):
        """
        x: torch FloatTensor (B,S,E)
            batch by seq length by embedding size
        x_mask: torch tensor (B,S)
            a mask to prevent some tokens from influencing the output
        """
        if x.shape[1] != self.seq_len:
            mask = self.get_mask(x.shape[1], self.state_size)
            mask = mask.cuda(non_blocking=True)
            self.register_buffer("mask",mask)
        else:
            mask = self.mask

        # TODO: figure out your fucking life and test this module
        reps = 1 if self.multi_init else self.state_size
        inits = self.inits.repeat(len(x),reps,1)
        x = torch.cat([x,inits],dim=1)
        if self.prob_embs:
            x = sample_probs(x)
        fx = self.pos_encoding(x)
        fx = self.dropout(fx)
        for i,enc in enumerate(self.enc_layers):
            fx = enc(fx,mask=mask,x_mask=x_mask)
            if self.prob_embs and i < len(self.enc_layers)-1:
                fx = sample_probs(fx)
        return fx

    def get_mask(self, seq_len, state_size, bidirectional=False):
        """
        Returns a mask that blocks the future vectors in the sequence
        but does not block the state vectors.
        """
        mask = np.tri(seq_len+state_size,seq_len+state_size)
        mask[:,seq_len:] = 1
        mask = torch.FloatTensor(mask)
        mask = mask.masked_fill(mask==0,-1e10)
        mask[mask==1] = 0
        return mask

class TransformerBase(nn.Module):
    """
    This is a class to hold all of the key parameters argued to the
    transformer. This makes it easier to create variations of the
    transformer architecture.
    """
    def __init__(self, seq_len=None, n_vocab=None,
                                     emb_size=512,
                                     enc_slen=None,
                                     dec_slen=None,
                                     attn_size=64,
                                     n_heads=8,
                                     act_fxn="ReLU",
                                     enc_layers=3,
                                     dec_layers=3,
                                     enc_mask=False,
                                     class_h_size=4000,
                                     class_bnorm=True,
                                     class_drop_p=0,
                                     enc_drop_p=0,
                                     dec_drop_p=0,
                                     ordered_preds=False,
                                     init_decs=False,
                                     idx_inputs=True,
                                     idx_outputs=True,
                                     prob_embs=False,
                                     prob_attn=False,
                                     mask_idx=0,
                                     start_idx=None,
                                     stop_idx=None,
                                     dec_mask_idx=0,
                                     multi_init=False,
                                     use_dec_embs=False,
                                     n_vocab_out=None,
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
        act_fxn: str
            the activation function to be used in the MLPs
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
        enc_drop_ps: float or list of floats
            the dropout probability for each encoding layer
        dec_drop_ps: float or list of floats
            the dropout probability for each decoding layer
        ordered_preds: bool
            if true, the decoder will mask the predicted sequence so
            that the attention modules will not see the tokens ahead
            located further along in the sequence.
        init_decs: bool
            if true, an initialization decoding vector is learned as
            the initial input to the decoder.
        idx_inputs: bool
            if true, the inputs are integer (long) indexes that require
            an embedding layer. Otherwise it is assumed that the inputs
            are feature vectors that do not require an embedding layer
        idx_outputs: bool
            if true, the output sequence (y) is integer (long) indexes
            that require an embedding layer. Otherwise it is assumed
            that the outputs are feature vectors that do not require
            an embedding layer
        prob_embs: bool
            if true, all embedding vectors are treated as parameter
            vectors for gaussian distributions before being fed into
            the transformer architecture
        prob_attn: bool
            if true, the queries and keys are projected into a mu and
            sigma vector and sampled from a gaussian distribution
            before the attn mechanism
        mask_idx: int
            the numeric index of the mask token
        start_idx: int
            the numeric index of the start token
        stop_idx: int
            the numeric index of the stop token
        dec_mask_idx: int
            the numeric index of the mask token for the decoding tokens
        multi_init: bool
            if true, the initialization vector has a unique value for
            each slot in the generated sequence
        use_dec_embs: bool
            determines if the decoding inputs use a unique embedding
            layer. Defaults to true if n_vocab_out is not None and is
            different than n_vocab.
        n_vocab_out: int or None
            the number of potential output words. Defaults to n_vocab
            if None is argued. Additionally, if n_vocab_out is not None,
            a new embedding is created for the decoding inputs.
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
        self.init_decs = init_decs
        self.idx_inputs = idx_inputs
        self.idx_outputs = idx_outputs
        self.prob_embs = prob_embs
        self.prob_attn = prob_attn
        self.mask_idx = mask_idx
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.dec_mask_idx = dec_mask_idx
        self.multi_init = multi_init
        self.use_dec_embs = use_dec_embs
        if n_vocab_out is None:
            self.n_vocab_out = self.n_vocab
        else:
            self.use_dec_embs = use_dec_embs or n_vocab_out != n_vocab
            self.n_vocab_out = n_vocab_out

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
        print("enc_slen:", self.enc_slen)
        print("dec_slen:", self.dec_slen)

        # Encoding Embedding Layer
        # If we want to share the embedding but we don't want to index
        # the inputs, we still want to create an embedding. But, if we
        # dont want to index the outputs either, then we don't care
        # if we want to share the embeddings or not
        temp = (not self.use_dec_embs and not self.idx_outputs)
        if not self.idx_inputs and temp:
            self.embeddings = None
        elif self.prob_embs: 
            self.embeddings =nn.Embedding(self.n_vocab,2*self.emb_size)
        else:
            self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        # Decoding Embedding Layer
        if self.use_dec_embs and not self.idx_outputs:
            self.dec_embeddings = None
        elif not self.use_dec_embs:
            self.dec_embeddings = self.embeddings
        elif self.prob_embs:
            self.dec_embeddings =nn.Embedding(self.n_vocab_out,
                                              2*self.emb_size)
        else:
            self.dec_embeddings = nn.Embedding(self.n_vocab_out,
                                               self.emb_size)

        self.encoder = Encoder(self.enc_slen,emb_size=self.emb_size,
                                            attn_size=self.attn_size,
                                            n_layers=self.enc_layers,
                                            n_heads=self.n_heads,
                                            use_mask=self.enc_mask,
                                            act_fxn=self.act_fxn,
                                            prob_attn=self.prob_attn,
                                            prob_embs=self.prob_embs,
                                            drop_p=self.enc_drop_p)

        self.use_mask = not self.init_decs and self.ordered_preds
        if self.init_decs:
            print("init_decs prevented use of mask in decoder")
        self.decoder = Decoder(self.dec_slen,self.emb_size,
                                            self.attn_size,
                                            self.dec_layers,
                                            n_heads=self.n_heads,
                                            act_fxn=self.act_fxn,
                                            use_mask=self.use_mask,
                                            init_decs=self.init_decs,
                                            multi_init=self.multi_init,
                                            prob_attn=self.prob_attn,
                                            prob_embs=self.prob_embs,
                                            drop_p=self.dec_drop_p)

        self.classifier = Classifier(self.emb_size,
                                     self.n_vocab_out,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)

    def forward(self, x, y, x_mask=None, y_mask=None, ret_latns=False):
        """
        x: float tensor (B,S)
        y: float tensor (B,S)
        x_mask: float tensor (B,S)
            defaults to indexes equal to 0 if None is argued
        ret_latns: bool
            if true, the latent decodings are returned in addition to
            the classification predictions
        """
        if self.idx_inputs is not None and self.idx_inputs:
            embs = self.embeddings(x)
            if x_mask is None:
                bitmask = (x==self.mask_idx)
                x_mask = bitmask.float().masked_fill(bitmask,-1e10)
        else:
            embs = x
        encs = self.encoder(embs,x_mask=x_mask)
        # Teacher force
        if self.training or not self.use_mask:
            return self.teacher_force_fwd(encs,y,x_mask,y_mask,
                                                        ret_latns)
        else:
            latns = []
            pred_distr = []
            outputs = y[:,:1]
            if self.idx_outputs and y_mask is None:
                bitmask = (y==self.dec_mask_idx)
                y_mask = bitmask.masked_fill(bitmask,-1e10)
            for i in range(y.shape[1]):
                if self.idx_outputs:
                    dembs = self.dec_embeddings(outputs)
                    temp_mask = y_mask[:,:outputs.shape[1]]
                else:
                    dembs = outputs
                    if y_mask is not None:
                        temp_mask = y_mask[:,:outputs.shape[1]]
                    else:
                        temp_mask = y_mask
                decs = self.decoder(dembs, encs, x_mask=temp_mask,
                                                 enc_mask=x_mask)
                decs = decs[:,-1:] # only take the latest element
                decs = decs.reshape(-1,decs.shape[-1])
                preds = self.classifier(decs)
                preds = preds.reshape(len(y),1,preds.shape[-1])
                pred_distr.append(preds)
                if self.idx_outputs:
                    argmaxs = torch.argmax(preds,dim=-1)
                else: argmaxs = decs
                outputs = torch.cat([outputs,argmaxs],dim=1)
            preds = torch.cat(pred_distr,dim=1)
            if ret_latns:
                latns = decs.reshape(len(y),y.shape[1],decs.shape[-1])
                return preds,latns
            return preds

    def teacher_force_fwd(self, encs, y, x_mask, y_mask, ret_latns):
        """
        Use this only as an offshoot of the forward function.
        Uses teacher forcing to train the model.
        """
        if self.idx_outputs:
            dembs = self.dec_embeddings(y)
            if y_mask is None:
                bitmask = (y==self.dec_mask_idx)
                y_mask = bitmask.masked_fill(bitmask,-1e10)
        else:
            dembs = y
        decs = self.decoder(dembs, encs, x_mask=y_mask,
                                         enc_mask=x_mask)
        decs = decs.reshape(-1,decs.shape[-1])
        preds = self.classifier(decs)
        preds = preds.reshape(len(y),y.shape[1],preds.shape[-1])
        if ret_latns:
            return preds,decs.reshape(len(y),y.shape[1],decs.shape[-1])
        return preds

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

