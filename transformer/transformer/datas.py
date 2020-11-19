import bcolz
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import copy
import pickle
import tokenizer as tk
import random
import skimage.io
from transformer.utils import get_max_key
from tokenizers import CharBPETokenizer
import ml_utils.datas
import contextlib
import time

class EngGerDataset(Dataset):
    """
    Can be english to german or german to english.
    """
    def __init__(self, data_folder, rank=0,
                                    val_set=False,
                                    world_size=1,
                                    seed=0,
                                    eng_to_ger=True,
                                    vocab_size=37000,
                                    MASK="<MASK>",
                                    START="<START>",
                                    STOP="<STOP>",
                                    exp_name="",
                                    max_context=None,
                                    batch_size=128,
                                    val_size=30000,
                                    **kwargs):
        """
        rank: int
            the rank in the distributed training
        val_set: bool
            if true, this dataset is created as the validation set
        world_size: int
            the number of processes if using distributed training
        seed: int
            random seed
        data_folder: str
            the path to the folder that should contain a `train.en` and
            a `train.de` file.
        eng_to_ger: bool
            if true, the x values are returned as english ids and the
            y values are german ids. If false, then visa-versa
        vocab_size: int
            the number of encodings for the byte-pair encoding scheme
        MASK: str
            the mask token
        START: str
            the start token
        STOP: str
            the stop token
        exp_name: str
            name of the experiment
        max_context: int
            the maximum sequence length
        val_size: int
            the number of samples to be set aside for validation
        """
        self.rank = rank
        print("rank:", self.rank)
        self.world_size = world_size
        self.val_set = val_set
        self.val_size = val_size
        self.batch_size = batch_size
        self.data_folder = os.path.expanduser(data_folder)
        self.en_path = os.path.join(data_folder, "train.en")
        self.de_path = os.path.join(data_folder, "train.de")
        self.eng_to_ger = eng_to_ger
        self.vocab_size = vocab_size
        self.MASK = MASK
        self.START = START
        self.STOP = STOP
        self.max_context = max_context
        self.en_tok_path = os.path.join(self.data_folder,"en_tokenizer")
        self.de_tok_path = os.path.join(self.data_folder,"de_tokenizer")
        self.en_arr_path = os.path.join(self.data_folder,"en_bcolz")
        self.de_arr_path = os.path.join(self.data_folder,"de_bcolz")
        self.en_lens_path = os.path.join(self.data_folder,"en_bcolz_lens")
        self.de_lens_path = os.path.join(self.data_folder,"de_bcolz_lens")

        # Train tokenizers
        if rank==0: print("Tokenizing english..")
        self.en_tokenizer = CharBPETokenizer()
        if os.path.exists(self.en_tok_path): # Load trained tokenizer
            if rank==0:
                print("loading from pretrained tokenizer",
                                         self.en_tok_path)
            self.en_tokenizer = ml_utils.datas.load_tokenizer(
                                               self.en_tokenizer,
                                               self.en_tok_path)
        else:
            self.en_tokenizer.train([self.en_path],vocab_size=vocab_size)
            os.mkdir(self.en_tok_path)
            self.en_tokenizer.save_model(self.en_tok_path)
        self.en_tokenizer.add_special_tokens([self.MASK])
        self.en_tokenizer.add_tokens([self.START])
        self.en_tokenizer.add_tokens([self.STOP])
        self.en_mask_idx = self.en_tokenizer.token_to_id(self.MASK)
        self.en_start_idx = self.en_tokenizer.token_to_id(self.START)
        self.en_stop_idx = self.en_tokenizer.token_to_id(self.STOP)

        if rank==0: print("Tokenizing german..")
        self.de_tokenizer = CharBPETokenizer()
        if os.path.exists(self.de_tok_path): # Load trained tokenizer
            if rank==0:
                print("loading from pretrained tokenizer",
                                         self.de_tok_path)
            self.de_tokenizer = ml_utils.datas.load_tokenizer(
                                               self.de_tokenizer,
                                               self.de_tok_path)
        else:
            self.de_tokenizer.train([self.de_path],vocab_size=vocab_size)
            os.mkdir(self.de_tok_path)
            self.de_tokenizer.save_model(self.de_tok_path)
        self.de_tokenizer.add_special_tokens([self.MASK])
        self.de_tokenizer.add_tokens([self.START])
        self.de_tokenizer.add_tokens([self.STOP])
        self.de_mask_idx = self.de_tokenizer.token_to_id(self.MASK)
        self.de_start_idx = self.de_tokenizer.token_to_id(self.START)
        self.de_stop_idx = self.de_tokenizer.token_to_id(self.STOP)

        # Get English sentence lists
        if rank==0: print("Making english idxs")
        if os.path.exists(self.en_arr_path):
            if rank==0: print("loading from bcolz", self.en_arr_path)
            self.en_idxs = bcolz.carray(rootdir=self.en_arr_path)
            self.en_lens = bcolz.carray(rootdir=self.en_lens_path)
            self.en_max_len = self.en_idxs.shape[-1]
            if exp_name == "test":
                self.val_size = 250
                self.en_idxs = self.en_idxs[:1000]
                self.en_lens = self.en_lens[:1000]
            if self.world_size > 1:
                with temp_seed(seed-rank): 
                    sample_perm=np.random.permutation(len(self.en_idxs))
                if not self.val_set:
                    n_samps = (len(self.en_idxs)-self.val_size)
                    n_samps = n_samps//self.world_size
                    indices = sample_perm[rank*n_samps:(rank+1)*n_samps]
                else:
                    indices = sample_perm[-self.val_size:]
                try:
                    if rank==0:
                        print("splitting dataset.. ",end="")
                        starttime = time.time()
                    self.en_idxs = self.en_idxs[indices]
                    self.en_lens = self.en_lens[indices]
                    if rank==0: print("duration:", time.time()-starttime)
                except:
                    temp_idxs = []
                    temp_lens = []
                    if rank==0:
                        print("Collecting data")
                        rnge = tqdm(indices)
                    else: rnge = indices
                    for i in rnge:
                        temp_idxs.append(self.en_idxs[i])
                        temp_lens.append(self.en_lens[i])
                    self.en_idxs = np.asarray(temp_idxs)
                    self.en_lens = np.asarray(temp_lens)
                    if rank==0: print("duration:", time.time()-starttime)
        else:
            self.en_max_len = 0
            self.en_idxs = []
            self.en_lens = []
            with open(self.en_path, 'r') as f:
                for i,l in tqdm(enumerate(f.readlines())):
                    l = l.strip()
                    if len(l) > 0:
                        output = self.en_tokenizer.encode(l)
                        ids = [self.en_start_idx]+list(output.ids)\
                                                 +[self.en_stop_idx]
                        self.en_idxs.append(ids)
                        self.en_lens.append(len(ids))
                        if len(ids) > self.en_max_len:
                            self.en_max_len = len(ids)
                    if exp_name == "test" and i > 100:
                        break
            mask = [self.en_mask_idx for i in range(self.en_max_len)]
            l = 0
            if rank==0: print("Padding english idxs")
            for i in tqdm(range(len(self.en_idxs))):
                diff = self.en_max_len - len(self.en_idxs[i])
                self.en_idxs[i] = self.en_idxs[i] + mask[:diff]
            if rank==0: print("Saving to bcolz")
            self.en_idxs = bcolz.carray(self.en_idxs,
                                        rootdir=self.en_arr_path,
                                        dtype="int32")
            self.en_idxs.flush()
            self.en_lens = bcolz.carray(self.en_lens,
                                        rootdir=self.en_lens_path,
                                        dtype="int32")
            self.en_lens.flush()
        if self.en_max_len > max_context:
            if rank==0:
                print("Truncating context from", self.en_max_len,
                                          "to", self.max_context)
            self.en_max_len = self.max_context

        # Get German Sentence Lists
        if rank==0: print("Making german idxs")
        if os.path.exists(self.de_arr_path):
            if rank==0: print("loading from bcolz", self.de_arr_path)
            self.de_idxs = bcolz.carray(rootdir=self.de_arr_path)
            self.de_lens = bcolz.carray(rootdir=self.de_lens_path)
            self.de_max_len = self.de_idxs.shape[-1]
            if exp_name == "test":
                self.val_size = 250
                self.en_idxs = self.en_idxs[:1000]
                self.en_lens = self.en_lens[:1000]
            if self.world_size > 1:
                try:
                    if rank==0:
                        print("splitting dataset.. ",end="")
                        starttime = time.time()
                    self.de_idxs = self.de_idxs[indices]
                    self.de_lens = self.de_lens[indices]
                    if rank==0: print("duration:", time.time()-starttime)
                except:
                    temp_idxs = []
                    temp_lens = []
                    try:
                        if rank==0: print("Collecting data")
                        for i in rnge:
                            temp_idxs.append(self.de_idxs[i])
                            temp_lens.append(self.de_lens[i])
                    except Exception as e:
                        print("Likely error caused by bcolz existing "+\
                                               "for en but not de data")
                        print(e)
                        assert False
                    self.de_idxs = np.asarray(temp_idxs)
                    self.de_lens = np.asarray(temp_lens)
                    if rank==0: print("duration:", time.time()-starttime)
        else:
            self.de_max_len = 0
            self.de_idxs = []
            self.de_lens = []
            with open(self.de_path, 'r') as f:
                for i,l in tqdm(enumerate(f.readlines())):
                    l = l.strip()
                    if len(l) > 0:
                        output = self.de_tokenizer.encode(l)
                        ids = [self.de_start_idx]+list(output.ids)\
                                                 +[self.de_stop_idx]
                        self.de_idxs.append(ids)
                        self.de_lens.append(len(ids))
                        if len(ids) > self.de_max_len:
                            self.de_max_len = len(ids)
                    if exp_name == "test" and i > 100:
                        break
            mask = [self.de_mask_idx for i in range(self.de_max_len)]
            if rank==0: print("Padding german idxs")
            for i in tqdm(range(len(self.de_idxs))):
                diff = self.de_max_len - len(self.de_idxs[i])
                self.de_idxs[i] = self.de_idxs[i] + mask[:diff]
            if rank==0: print("Saving to bcolz")
            self.de_idxs = bcolz.carray(self.de_idxs,
                                        rootdir=self.de_arr_path,
                                        dtype="int32")
            self.de_idxs.flush()
            self.de_lens = bcolz.carray(self.de_lens,
                                        rootdir=self.de_lens_path,
                                        dtype="int32")
            self.de_lens.flush()
        if self.de_max_len > max_context:
            if rank==0:
                print("Truncating context from", self.de_max_len,
                                       "to", self.max_context)
            self.de_max_len = self.max_context

        if rank==0: print("Converting to numpy arrays")
        if self.eng_to_ger:
            self.X = np.asarray(self.en_idxs)
            self.X_lens = np.asarray(self.en_lens)
            self.X_tokenizer = self.en_tokenizer
            self.X_mask_idx = self.en_mask_idx
            self.X_start_idx = self.en_start_idx
            self.X_stop_idx = self.en_stop_idx
            self.X_max_len = self.en_max_len

            self.Y = np.asarray(self.de_idxs)
            self.Y_lens = np.asarray(self.de_lens)
            self.Y_tokenizer = self.de_tokenizer
            self.Y_mask_idx = self.de_mask_idx
            self.Y_start_idx = self.de_start_idx
            self.Y_stop_idx = self.de_stop_idx
            self.Y_max_len = self.de_max_len
        else:
            self.X = np.asarray(self.de_idxs)
            self.X_lens = np.asarray(self.de_lens)
            self.X_tokenizer = self.de_tokenizer
            self.X_mask_idx = self.de_mask_idx
            self.X_start_idx = self.de_start_idx
            self.X_stop_idx = self.de_stop_idx
            self.X_max_len = self.de_max_len

            self.Y = np.asarray(self.en_idxs)
            self.Y_lens = np.asarray(self.en_lens)
            self.Y_tokenizer = self.en_tokenizer
            self.Y_mask_idx = self.en_mask_idx
            self.Y_start_idx = self.en_start_idx
            self.Y_stop_idx = self.en_stop_idx
            self.Y_max_len = self.en_max_len

    def __len__(self):
        return len(self.en_idxs)
    
    def __getitem__(self,i,l=None):
        if l is None:
            l = self.X_lens[int(i)]
        idxs = []
        margin = 5
        while len(idxs)<25 and margin < 400:
            min_l = l-margin
            max_l = l+margin
            idxs = (self.X_lens>min_l)&(self.X_lens<max_l)
            margin += 5
        max_l = min(np.max(self.X_lens[idxs]),self.max_context)
        if max_l <   50 : batch_size = self.batch_size
        elif max_l < 70: batch_size = self.batch_size//2
        elif max_l < 100: batch_size = self.batch_size//4
        elif max_l < 120: batch_size = self.batch_size//8
        elif max_l < 140: batch_size = self.batch_size//16
        elif max_l < 160: batch_size = self.batch_size//32
        else: batch_size = self.batch_size//64
        batch_size = max(16,batch_size)
        perm = np.random.permutation(len(idxs))
        x = np.asarray(self.X[perm[:batch_size],:max_l])
        y = np.asarray(self.Y[perm[:batch_size],:max_l])
        return torch.LongTensor(x), torch.LongTensor(y)

    def get_largest_batch(self, size_num):
        l = 10
        if size_num == 1:
            l = 25
        elif size_num == 2:
            l = 400
        elif size_num == 3:
            l = 130
        elif size_num == 4:
            l = 75
        elif size_num == 5:
            l = 44
        elif size_num == 6:
            l = 94
        elif size_num == 7:
            l = 200
        elif size_num == 8:
            l = 300
        return self.__getitem__(0,l)

    def X_idxs2tokens(self, idxs):
        """
        idxs: LongTensor (N,)
            converts an array of tokens to a sentence
        """
        return self.X_tokenizer.decode(idxs)

    def Y_idxs2tokens(self, idxs):
        """
        idxs: LongTensor (N,)
            converts an array of tokens to a sentence
        """
        return self.Y_tokenizer.decode(idxs)

class VariableLengthSeqLoader:
    def __init__(self, dataset, samples_per_epoch=1000, shuffle=True,
                                                        **kwargs):
        """
        dataset: torch Dataset
        samples_per_epoch: int
        shuffle: bool
            if true, samples are drawn iid from dataset
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.perm = np.random.permutation(len(self.dataset))
        else:
            self.perm = np.arange(self.samples_per_epoch)
        self.perm = self.perm.astype(np.int)
        self.perm = self.perm[:self.samples_per_epoch]
        self.perm_idx = 0
        return self

    def __next__(self):
        if self.perm_idx >= self.samples_per_epoch:
            raise StopIteration
        i = self.perm[self.perm_idx]
        self.perm_idx += 1
        return self.dataset[i]

    def __len__(self):
        return self.samples_per_epoch

    def get_largest_batch(self, b):
        return self.dataset.get_largest_batch(b)

class DatasetWrapper(Dataset):
    """
    Used as a wrapper class to more easily split a dataset into a
    validation and training set. Simply create a training and validation
    set of indexes. Then argue the original dataset and the corresponding
    indexes to this class to create each the train and validation sets.
    """
    def __init__(self,dataset,idxs):
        """
        dataset: torch Dataset
        idxs: torch LongTensor or list of ints
        """
        self.dataset = dataset
        for attr in dir(dataset):
            if "__"!=attr[:2]:
                try:
                    setattr(self, attr, getattr(dataset,attr))
                except:
                    pass
        self.idxs = idxs
        assert len(self.idxs) <= len(self.dataset)
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        return self.dataset[idx]

def get_data(hyps):
    dataset = hyps['dataset']
    train_data = globals()[dataset](**hyps)
    val_data = None
    if hyps['rank']==0:
        val_data = globals()[dataset](val_set=True,**hyps)
    return train_data,val_data

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
