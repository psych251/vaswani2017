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

class EngGerDataset(Dataset):
    """
    Can be english to german or german to english.
    """
    def __init__(self, data_path, eng_to_ger=True, vocab_size=50000,
                                                   MASK="<MASK>",
                                                   **kwargs):
        """
        data_path: str
            the path to the folder that should contain a `train.en` and
            a `train.de` file.
        eng_to_ger: bool
            if true, the x values are returned as english ids and the
            y values are german ids. If false, then visa-versa
        vocab_size: int
            the number of encodings for the byte-pair encoding scheme
        MASK: str
            the mask token
        """
        self.data_path = os.path.expanduser(data_path)
        self.en_path = os.path.join(data_path, "train.en")
        self.de_path = os.path.join(data_path, "train.de")
        self.eng_to_ger = eng_to_ger
        self.MASK = MASK

        # Train tokenizers
        print("Tokenizing english..")
        self.en_tokenizer = CharBPETokenizer()
        self.en_tokenizer.train(self.en_path,vocab_size=vocab_size)
        self.en_tokenizer.add_special_tokens([self.MASK])
        self.en_mask_idx = self.en_tokenzier.token_to_id(self.MASK)
        print("Tokenizing german..")
        self.de_tokenizer = CharBPETokenizer()
        self.de_tokenizer.train(self.de_path,vocab_size=vocab_size)
        self.de_tokenizer.add_special_tokens([self.MASK])
        self.de_mask_idx = self.de_tokenzier.tokde_to_id(self.MASK)

        # Get English sentence lists
        self.en_max_len = 0
        self.en_idxs = []
        with open(self.en_path, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                if len(l) > 0:
                    output = self.en_tokenizer.encode(l)
                    self.en_idxs.append(output.ids)
                    if len(output.ids) > self.en_max_len:
                        self.en_max_len = len(output.ids)
        mask = [self.en_mask_idx for i in range(self.en_max_len)]
        for i in range(len(self.en_idxs)):
            diff = self.en_max_len - len(self.en_idxs[i])
            self.en_idxs[i] = self.en_idxs[i] + mask[:diff]
        self.en_idxs = torch.LongTensor(self.en_idxs)

        # Get German Sentence Lists
        self.de_max_len = 0
        self.de_idxs = []
        with open(self.de_path, 'r') as f:
            for l in f.readlines():
                l = l.strip()
                if len(l) > 0:
                    output = self.de_tokenizer.encode(l)
                    self.de_idxs.append(output.ids)
                    if len(output.ids) > self.de_max_len:
                        self.de_max_len = len(output.ids)
        mask = [self.de_mask_idx for i in range(self.de_max_len)]
        for i in range(len(self.de_idxs)):
            diff = self.de_max_len - len(self.de_idxs[i])
            self.de_idxs[i] = self.de_idxs[i] + mask[:diff]
        self.de_idxs = torch.LongTensor(self.de_idxs)

        self.X = self.en_idxs if self.eng_to_ger else self.de_idxs
        self.Y = self.de_idxs if self.eng_to_ger else self.en_idxs

    def __len__(self):
        return len(self.en_idxs)
    
    def __getitem__(self,i):
        return self.en_idxs[i],self.de_idxs[i]

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

