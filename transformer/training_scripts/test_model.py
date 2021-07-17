import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import time
from tqdm import tqdm
import math
from queue import Queue
from collections import deque
import psutil
import json
import transformer.datas as datas
import transformer.custom_modules as custmods
from transformer.models import *
import ml_utils.save_io as io
from ml_utils.training import get_exp_num, record_session, get_save_folder,get_resume_checkpt
from ml_utils.utils import try_key
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from nltk.translate.bleu_score import corpus_bleu
import sys


MASK  = "<MASK>"
START = "<START>"
STOP  = "<STOP>"

checkpt = io.load_checkpoint(sys.argv[1])
hyps = checkpt['hyps']
model = io.load_model(sys.argv[1], globals())
model.cuda()

# Load data:
data = datas.EngGerNewstest(**hyps)
X = []
Y = []
for i in range(len(data)):
    x,y = data[i]
    X.append(x)
    Y.append(y)
X = torch.stack(X,axis=0)
Y = torch.stack(Y,axis=0)
print("X", X.shape)
print("Y", Y.shape)
T = Y[:,1:]
Y = Y[:,:-1]
batch_size = 16
avg_bleu = 0
avg_loss = 0
avg_acc = 0
avg_indy_acc = 0
print("\nValidating...")
torch.cuda.empty_cache()
mask_idx = data.Y_mask_idx
stop_idx = data.Y_stop_idx
lossfxn = nn.CrossEntropyLoss()
n_loops = len(X)//batch_size
with torch.no_grad():
    for b in range(n_loops):
        startx = b*batch_size
        endx = startx+batch_size
        print("nLoops:", n_loops)
        print("b:", b)
        print("start:", startx)
        print("end:", endx)
        x = X[startx:endx]
        y = Y[startx:endx]
        targs = T[startx:endx]
        og_shape = targs.shape
        preds = model(x.cuda(), y.cuda())

        # Tot loss and acc
        preds = preds.reshape(-1,preds.shape[-1])
        targs = targs.reshape(-1).cuda(non_blocking=True)
        bitmask = targs!=mask_idx
        loss = lossfxn(preds[bitmask],targs[bitmask])
        if hyps['multi_gpu']:
            loss = loss.mean()
        preds = torch.argmax(preds,dim=-1)
        sl = int(og_shape[-1])
        eq = (preds==targs).float()
        indy_acc = eq[bitmask].mean()
        eq[~bitmask] = 1
        eq = eq.reshape(og_shape)
        acc = (eq.sum(-1)==sl).float().mean()
        bleu_trgs=targs.reshape(og_shape).data.cpu().numpy()
        trg_ends = np.argmax((bleu_trgs==stop_idx),axis=1)
        bleu_prds=preds.reshape(og_shape).data.cpu().numpy()
        prd_ends = np.argmax((bleu_prds==stop_idx),axis=1)
        btrgs = []
        bprds = []
        for i in range(len(bleu_trgs)):
            temp = bleu_trgs[i,None,:trg_ends[i]].tolist()
            btrgs.append(temp)
            bprds.append(bleu_prds[i,:prd_ends[i]].tolist())
        bleu = corpus_bleu(btrgs,bprds)
        avg_bleu += bleu
        avg_acc += acc.item()
        avg_indy_acc += indy_acc.item()
        avg_loss += loss.item()

        s="Loss:{:.5f} | Acc:{:.5f} | Bleu:{:.5f} | {:.0f}%"
        s = s.format(loss.item(), acc.item(), bleu,
                                  b/len(X)*100)
        print(s, end=len(s)*" " + "\r")
        if hyps['exp_name']=="test" and b > 5: break

val_avg_bleu = avg_bleu/n_loops
val_avg_loss = avg_loss/n_loops
val_avg_acc = avg_acc/n_loops
val_avg_indy = avg_indy_acc/n_loops
stats_string = "Val- Loss:{:.5f} | Acc:{:.5f} | "
stats_string += "Indy:{:.5f}\nVal Bleu: {:.5f}\n"
stats_string = stats_string.format(val_avg_loss,val_avg_acc,
                                                val_avg_indy,
                                                val_avg_bleu)
print(stats_string)

