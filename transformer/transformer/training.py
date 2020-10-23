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
import transformer.models as models
import ml_utils.save_io as io
from ml_utils.training import get_exp_num, record_session, get_save_folder,get_resume_checkpt
from ml_utils.utils import try_key

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

MASK  = "<MASK>"
START = "<START>"
STOP  = "<STOP>"

def train(hyps, verbose=True):
    """
    hyps: dict
        contains all relavent hyperparameters
    """
    hyps['main_path'] = try_key(hyps,'main_path',"./")
    checkpt,hyps = get_resume_checkpt(hyps)
    if checkpt is None:
        hyps['exp_num']=get_exp_num(hyps['main_path'], hyps['exp_name'])
        hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    # Set manual seed
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])
    hyps['MASK'] = MASK
    hyps['START'] = START
    hyps['STOP'] = STOP

    model_class = hyps['model_class']
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)
    if not hyps['init_decs'] and not hyps['ordered_preds']:
        s = "WARNING!! You probably want to set ordered preds to True "
        s += "with your current configuration!!"
        print(s)

    if verbose:
        print("Retreiving Dataset")
    if "shuffle_split" not in hyps and hyps['shuffle']:
        hyps['shuffle_split'] = True
    train_data,val_data = datas.get_data(**hyps)

    hyps['enc_slen'] = train_data.X.shape[-1]
    hyps['dec_slen'] = train_data.Y.shape[-1]
    hyps["mask_idx"] = train_data.X_tokenizer.token_to_id(MASK)
    hyps["dec_mask_idx"] = train_data.Y_tokenizer.token_to_id(MASK)
    hyps['n_vocab'] = train_data.X_tokenizer.get_vocab_size()
    hyps['n_vocab_out'] = train_data.Y_tokenizer.get_vocab_size()
    train_loader = torch.utils.data.DataLoader(train_data,
                                    batch_size=hyps['batch_size'],
                                    shuffle=hyps['shuffle'])
    val_loader = torch.utils.data.DataLoader(val_data,
                                    batch_size=hyps['batch_size'])

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)
    if try_key(hyps,"multi_gpu",False):
        ids = [i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=ids)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    # Load State Dicts if Resuming Training
    if checkpt is not None:
        if verbose:
            print("Loading state dicts from", checkpt['save_folder'])
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
    lossfxn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    if verbose:
        print("Beginning training for {}".format(hyps['save_folder']))
        print("train shape:", (len(train_data),*train_data.X.shape[1:]))
        print("val shape:", (len(val_data),*val_data.X.shape[1:]))
    record_session(hyps,model)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    epoch = -1
    mask_idx = train_data.Y_mask_idx
    print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch, hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        model.train()
        print("Training...")
        optimizer.zero_grad()
        for b,(x,y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            targs = y.data[:,1:]
            og_shape = targs.shape
            y = y[:,:-1]
            logits = model(x.to(DEVICE), y.to(DEVICE))
            preds = torch.argmax(logits,dim=-1)

            if epoch % 3 == 0 and b == 0:
                inp = x[0].data.cpu().numpy()
                trg = targs[0].data.numpy()
                prd = preds[0].data.cpu().numpy()
                print("Inp:", train_data.X_idxs2tokens(inp))
                print("Targ:", train_data.Y_idxs2tokens(trg))
                print("Pred:", train_data.Y_idxs2tokens(prd))

            # Tot loss
            logits = logits.reshape(-1,logits.shape[-1])
            targs = targs.reshape(-1).to(DEVICE)
            bitmask = targs!=mask_idx
            loss = lossfxn(logits[bitmask],targs[bitmask])

            loss = loss/hyps['n_loss_loops']
            loss.backward()
            if b % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Acc
            preds = preds.reshape(-1)
            sl = og_shape[-1]
            eq = (preds==targs).float()
            indy_acc = eq[bitmask].mean()
            eq[~bitmask] = 1
            eq = eq.reshape(og_shape)
            acc = (eq.sum(-1)==sl).float().mean()
            avg_acc += acc.item()
            avg_indy_acc += indy_acc.item()
            avg_loss += loss.item()

            s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
            s = s.format(loss.item(), acc.item(),
                                      b/len(train_loader)*100)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and b>5: break

        print()
        optimizer.zero_grad()
        train_avg_loss = avg_loss/len(train_loader)
        train_avg_acc = avg_acc/len(train_loader)
        train_avg_indy = avg_indy_acc/len(train_loader)

        s = "Train - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_avg_acc,
                                                train_avg_indy)

        ###### VALIDATION
        model.eval()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        print("Validating...")
        torch.cuda.empty_cache()
        with torch.no_grad():
            rand_word_batch = int(np.random.randint(0,len(val_loader)))
            for b,(x,y) in enumerate(val_loader):
                targs = y.data[:,1:]
                og_shape = targs.shape
                y = y[:,:-1]
                preds = model(x.to(DEVICE), y.to(DEVICE))

                # Tot loss and acc
                preds = preds.reshape(-1,preds.shape[-1])
                targs = targs.reshape(-1).to(DEVICE)
                bitmask = targs!=mask_idx
                loss = lossfxn(preds[bitmask],targs[bitmask])
                preds = torch.argmax(preds,dim=-1)
                sl = og_shape[-1]
                eq = (preds==targs).float()
                indy_acc = eq[bitmask].mean()
                eq[~bitmask] = 1
                eq = eq.reshape(og_shape)
                acc = (eq.sum(-1)==sl).float().mean()
                avg_acc += acc.item()
                avg_indy_acc += indy_acc.item()
                avg_loss += loss.item()

                if b == rand_word_batch or hyps['exp_name']=="test":
                    rand = int(np.random.randint(0,len(x)))
                    inp = x.data[rand].cpu().numpy()
                    inp_samp = val_data.X_idxs2tokens(inp)
                    trg=targs.reshape(og_shape)[rand].data.cpu().numpy()
                    targ_samp = val_data.Y_idxs2tokens(trg)
                    prd=preds.reshape(og_shape)[rand].data.cpu().numpy()
                    pred_samp = val_data.Y_idxs2tokens(prd)
                s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                s = s.format(loss.item(), acc.item(),
                                          b/len(val_loader)*100)
                print(s, end=len(s)*" " + "\r")
                if hyps['exp_name']=="test" and b > 5: break


        print()
        val_avg_loss = avg_loss/len(val_loader)
        val_avg_acc = avg_acc/len(val_loader)
        scheduler.step(val_avg_acc)
        val_avg_indy = avg_indy_acc/len(val_loader)
        stats_string += "Val - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = stats_string.format(val_avg_loss,val_avg_acc,
                                                        val_avg_indy)
        stats_string += "Inp: "  + inp_samp  + "\n"
        stats_string += "Targ: " + targ_samp + "\n"
        stats_string += "Pred: " + pred_samp + "\n"
        optimizer.zero_grad()

        save_dict = {
            "epoch":epoch,
            "hyps":hyps,
            "train_loss":train_avg_loss,
            "train_acc":train_avg_acc,
            "train_indy":train_avg_indy,
            "val_loss":val_avg_loss,
            "val_acc":val_avg_acc,
            "val_indy":val_avg_indy,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
        }
        save_name = "checkpt"
        save_name = os.path.join(hyps['save_folder'],save_name)
        io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                del_prev_sd=hyps['del_prev_sd'])
        stats_string += "Exec time: {}\n".format(time.time()-starttime)
        print(stats_string)
        s = "Epoch:{} | Model:{}\n".format(epoch, hyps['save_folder'])
        stats_string = s + stats_string
        log_file = os.path.join(hyps['save_folder'],"training_log.txt")
        with open(log_file,'a') as f:
            f.write(str(stats_string)+'\n')
    del save_dict['state_dict']
    del save_dict['optim_dict']
    del save_dict['hyps']
    save_dict['save_folder'] = hyps['save_folder']
    return save_dict

def save_data_structs(hyps, structs):
    """
    Records a copy of the data structs that were used to create this
    dataset

    structs: list of samp_structs
        see the datas.WordProblem class for details on sample structs
    """
    sf = hyps['save_folder']
    if not os.path.exists(sf):
        os.mkdir(sf)
    h = "samp_structs.json"
    with open(os.path.join(sf,h),'w') as f:
        json.dump(structs, f)


def mask_words(x,y,mask_p=.15,mask_idx=0):
    """
    x: Long tensor (..., S)
        a torch tensor of token indices
    y: Long tensor (..., S)
        a torch tensor of token indices in which the sequence is
        offset forward by 1 position
    mask_p: float [0,1]
        the probability of masking a word
    mask_idx: int
        the index of the mask token
    """
    if mask_p == 0:
        return x,y,torch.zeros(x.shape).bool()
    probs = torch.rand(x.shape)
    mask = (probs<mask_p).bool()
    x[mask] = mask_idx
    postpender = torch.zeros(*x.shape[:-1],1).bool()
    mask = torch.cat([mask,postpender],dim=-1)[...,1:].bool()
    y[mask] = mask_idx
    return x,y,mask

