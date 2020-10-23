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

class EngGerDataset(Dataset):
    """
    Can be english to german or german to english.
    """
    def __init__(self, data_folder, eng_to_ger=True, vocab_size=50000,
                                                   MASK="<MASK>",
                                                   START="<START>",
                                                   STOP="<STOP>",
                                                   exp_name="",
                                                   **kwargs):
        """
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
        """
        self.data_folder = os.path.expanduser(data_folder)
        self.en_path = os.path.join(data_folder, "train.en")
        self.de_path = os.path.join(data_folder, "train.de")
        self.eng_to_ger = eng_to_ger
        self.MASK = MASK
        self.START = START
        self.STOP = STOP
        self.en_tok_path = os.path.join(self.data_folder,"en_tokenizer")
        self.de_tok_path = os.path.join(self.data_folder,"de_tokenizer")

        # Train tokenizers
        print("Tokenizing english..")
        self.en_tokenizer = CharBPETokenizer()
        if os.path.exists(self.en_tok_path): # Load trained tokenizer
            print("loading from pretrained tokenizer", self.en_tok_path)
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

        print("Tokenizing german..")
        self.de_tokenizer = CharBPETokenizer()
        if os.path.exists(self.de_tok_path): # Load trained tokenizer
            print("loading from pretrained tokenizer", self.de_tok_path)
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
        self.en_max_len = 0
        self.en_idxs = []
        with open(self.en_path, 'r') as f:
            for i,l in enumerate(f.readlines()):
                l = l.strip()
                if len(l) > 0:
                    output = self.en_tokenizer.encode(l)
                    ids = [self.en_start_idx]+list(output.ids)\
                                             +[self.en_stop_idx]
                    self.en_idxs.append(ids)
                    if len(ids) > self.en_max_len:
                        self.en_max_len = len(ids)
                if exp_name == "test" and i > 100:
                    break
        mask = [self.en_mask_idx for i in range(self.en_max_len)]
        l = 0
        for i in range(len(self.en_idxs)):
            diff = self.en_max_len - len(self.en_idxs[i])
            self.en_idxs[i] = self.en_idxs[i] + mask[:diff]
        self.en_idxs = torch.LongTensor(self.en_idxs)

        # Get German Sentence Lists
        self.de_max_len = 0
        self.de_idxs = []
        with open(self.de_path, 'r') as f:
            for i,l in enumerate(f.readlines()):
                l = l.strip()
                if len(l) > 0:
                    output = self.de_tokenizer.encode(l)
                    ids = [self.de_start_idx]+list(output.ids)\
                                             +[self.de_stop_idx]
                    self.de_idxs.append(ids)
                    if len(ids) > self.de_max_len:
                        self.de_max_len = len(ids)
                if exp_name == "test" and i > 100:
                    break
        mask = [self.de_mask_idx for i in range(self.de_max_len)]
        for i in range(len(self.de_idxs)):
            diff = self.de_max_len - len(self.de_idxs[i])
            self.de_idxs[i] = self.de_idxs[i] + mask[:diff]
        self.de_idxs = torch.LongTensor(self.de_idxs)

        if self.eng_to_ger:
            self.X = self.en_idxs
            self.X_tokenizer = self.en_tokenizer
            self.X_mask_idx = self.en_mask_idx
            self.X_start_idx = self.en_start_idx
            self.X_stop_idx = self.en_stop_idx

            self.Y = self.de_idxs
            self.Y_tokenizer = self.de_tokenizer
            self.Y_mask_idx = self.de_mask_idx
            self.Y_start_idx = self.de_start_idx
            self.Y_stop_idx = self.de_stop_idx
        else:
            self.X = self.de_idxs
            self.X_tokenizer = self.de_tokenizer
            self.X_mask_idx = self.de_mask_idx
            self.X_start_idx = self.de_start_idx
            self.X_stop_idx = self.de_stop_idx

            self.Y = self.en_idxs
            self.Y_tokenizer = self.en_tokenizer
            self.Y_mask_idx = self.en_mask_idx
            self.Y_start_idx = self.en_start_idx
            self.Y_stop_idx = self.en_stop_idx


    def __len__(self):
        return len(self.en_idxs)
    
    def __getitem__(self,i):
        return self.X[i],self.Y[i]

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
        try:
            idx = self.idxs[idx]
            return self.dataset[idx]
        except FileNotFoundError as e:
            while True:
                try:
                    idx = rand_sample(self.idxs)
                    return self.dataset[idx]
                except FileNotFoundError as e:
                    pass

def get_data(dataset, shuffle=True, **kwargs):
    dataset = globals()[dataset](**kwargs)
    if shuffle: perm = torch.randperm(len(dataset))
    else: perm = torch.arange(len(dataset))
    n_val = int(min(.2*len(dataset), 30000))
    val_idxs = perm[:n_val]
    train_idxs = perm[n_val:]
    val_data = DatasetWrapper(dataset, val_idxs)
    train_data = DatasetWrapper(dataset, train_idxs)
    return train_data,val_data
