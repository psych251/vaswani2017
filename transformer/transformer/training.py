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
from ml_utils.training import get_exp_num, record_session, get_save_folder
from ml_utils.utils import try_key

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def train(hyps, verbose=True):
    """
    hyps: dict
        contains all relavent hyperparameters
    """
    # Set manual seed
    hyps['exp_num'] = get_exp_num(hyps['main_path'], hyps['exp_name'])
    hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])
    model_class = hyps['model_class']
    hyps['model_type'] = custmods.TRANSFORMER_TYPE[model_class]
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)
    if not hyps['init_decs'] and not hyps['gen_decs'] and\
                                not hyps['ordered_preds']:
        s = "WARNING!! You probably want to set ordered preds to True "
        s += "with your current configuration!!"
        print(s)

    if verbose:
        print("Retreiving Dataset")
    if "shuffle_split" not in hyps and hyps['shuffle']:
        hyps['shuffle_split'] = True
    train_data,val_data = datas.get_data(**hyps)
    if not hasattr(train_data, "sampled_types"):
        setattr(train_data, "sampled_types", None)
        setattr(train_data, "samp_structs", None)
        setattr(val_data, "sampled_types", None)
        setattr(val_data, "samp_structs", None)
    hyps['enc_slen'] = train_data.X.shape[-1]
    hyps['dec_slen'] = train_data.Y.shape[-1] + 1
    tup = train_data[0]
    if len(tup) == 3: hyps["img_shape"] = tup[-1].shape
    elif len(tup) > 3: 
        hyps["img_shape"] = tup[0].shape
        hyps['count_len'] = tup[-1].shape[-1] + 1
    else: hyps['img_shape'] = tup[0].shape
    train_loader = torch.utils.data.DataLoader(train_data,
                                    batch_size=hyps['batch_size'],
                                    shuffle=hyps['shuffle'])
    bsize = hyps['batch_size'] if hyps['dataset']=="CLEVR" else 500
    val_loader = torch.utils.data.DataLoader(val_data,
                                    batch_size=hyps['batch_size'])
    hyps['n_vocab'] = len(train_data.word2idx.keys())
    tokz = train_data.tokenizer
    hyps['mask_idx']  = train_data.word2idx[tokz.MASK]
    hyps['start_idx'] = train_data.word2idx[tokz.START]
    hyps['stop_idx']  = train_data.word2idx[tokz.STOP]

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)
    if 'multi_gpu' in hyps and hyps['multi_gpu']:
        ids = [i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=ids)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    init_checkpt = try_key(hyps,"init_checkpt", None)
    if init_checkpt is not None and init_checkpt != "":
        if verbose:
            print("Loading state dicts from", init_checkpt)
        checkpt = io.load_checkpoint(init_checkpt)
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
        print("img shape:", hyps['img_shape'])

    record_session(hyps,model)
    if hyps['dataset'] == "WordProblem":
        save_data_structs(train_data.samp_structs)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    epoch = -1
    alpha = hyps['loss_alpha']
    mask_idx = train_data.word2idx["<MASK>"]
    null_loc = train_data.null_loc
    print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch, hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        avg_c_loss = 0
        avg_c_acc = 0
        avg_c_indy_acc = 0
        avg_z_loss = 0
        mask_avg_loss = 0
        mask_avg_acc = 0
        model.train()
        print("Training...")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        for b,tup in enumerate(train_loader):
            x,y = tup[:2]
            targs = y.data[:,1:]
            if hyps['init_decs']:
                y = train_data.inits.clone().repeat(len(targs),1)
            og_shape = targs.shape
            idx2word = train_data.idx2word
            if hyps['masking_task']:
                x,y,mask = mask_words(x, y, mask_p=hyps['mask_p'])
            y = y[:,:-1]

            z,c = None,None
            if len(tup) == 3:
                z = tup[-1].to(DEVICE)
            elif len(tup) == 4: 
                z,c = tup[-2].to(DEVICE), tup[-1].to(DEVICE)
                c_targs = c[:,1:]
                c_og_shape = c_targs.shape
                c = c[:,:-1]
                z_targs = z[:,1:]
                z_og_shape = z_targs.shape
                z = z[:,:-1]
            preds = model(x.to(DEVICE), y.to(DEVICE), z, c)
            c_preds,z_preds = None, None
            if isinstance(preds,tuple):
                preds,c_preds,z_preds = preds

            if epoch % 3 == 0 and b == 0:
                whr = torch.where(y[0]==mask_idx)[0]
                endx = y.shape[-1] if len(whr) == 0 else whr[0].item()
                print("y:",[idx2word[a.item()] for a in y[0,:endx]])
                print("t:",[idx2word[a.item()] for a in targs[0,:endx]])
                ms = torch.argmax(preds,dim=-1)
                print("p:", [idx2word[a.item()] for a in ms[0,:endx]])
                del ms
                if c_preds is not None:
                    mask = c_targs[0]!=mask_idx
                    l = [idx2word[a.item()] for a in c_targs[0][mask]]
                    print("ct:",l)
                    ms = torch.argmax(c_preds,dim=-1)
                    l = [idx2word[a.item()] for a in ms[0][mask]]
                    print("cp:", l)
                    del ms
                if z_preds is not None:
                    s = "{:.3f} "*10
                    s = s.format(*z[0][:,0].tolist()[:10])
                    print("ztx:", s)
                    s = "{:.3f} "*10
                    s = s.format(*z_preds[0][:,0].tolist()[:10])
                    print("zpx:", s)
                    s = "{:.3f} "*10
                    s = s.format(*z[0][:,1].tolist()[:10])
                    print("zty:", s)
                    s = "{:.3f} "*10
                    s = s.format(*z_preds[0][:,1].tolist()[:10])
                    print("zpy:", s)

            if hyps['masking_task']:
                print("masking!")
                # Mask loss and acc
                preds = preds.reshape(-1,preds.shape[-1])
                mask = mask.reshape(-1).bool()
                idxs = torch.arange(len(mask))[mask]
                mask_preds = preds[idxs]
                mask_targs = targs[idxs]
                mask_loss = lossfxn(mask_preds,mask_targs)
                mask_preds = torch.argmax(mask_preds,dim=-1)
                mask_acc = (mask_preds==mask_targs).sum().float()
                mask_acc = mask_acc/idxs.numel()

                mask_avg_acc  += mask_acc.item()
                mask_avg_loss += mask_loss.item()
            else:
                mask_loss = torch.zeros(1).to(DEVICE)
                mask_acc = torch.zeros(1).to(DEVICE)

                mask_avg_acc  += mask_acc.item()
                mask_avg_loss += mask_loss.item()

            # Tot loss
            preds = preds.reshape(-1,preds.shape[-1])
            targs = targs.reshape(-1).to(DEVICE)
            bitmask = targs!=mask_idx
            loss = lossfxn(preds[bitmask],targs[bitmask])
            if c_preds is not None:
                c_preds = c_preds.reshape(-1,c_preds.shape[-1])
                c_targs = c_targs.reshape(-1)
                c_bitmask = c_targs!=mask_idx
                c_loss = lossfxn(c_preds[c_bitmask],c_targs[c_bitmask])
            else: c_loss = torch.zeros(1).to(DEVICE)
            if z_preds is not None:
                z_targs = z_targs.to(DEVICE)
                z_bitmask = z_targs > null_loc
                z_loss = F.mse_loss(z_preds[z_bitmask],
                                    z_targs[z_bitmask])
                avg_z_loss += z_loss.item()
            else: z_loss = torch.zeros(1).to(DEVICE)

            if hyps['masking_task']:
                tot_loss = (alpha)*loss + (1-alpha)*mask_loss + c_loss
            else:
                tot_loss = loss + c_loss + z_loss
            tot_loss = tot_loss/hyps['n_loss_loops']
            tot_loss.backward()
            if b % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Acc
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
            if c_preds is not None:
                c_preds=torch.argmax(c_preds,dim=-1)
                sl = c_og_shape[-1]
                eq = (c_preds==c_targs).float()
                c_indy_acc = eq.reshape(-1)[c_bitmask].mean()
                eq[~c_bitmask] = 1
                eq = eq.reshape(c_og_shape)
                c_acc = (eq.sum(-1)==sl).float().mean()
                avg_c_acc += c_acc.item()
                avg_c_indy_acc += c_indy_acc.item()
                avg_c_loss += c_loss.item()

            if hyps["masking_task"]:
                s = "Mask Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                s = s.format(mask_loss.item(), mask_acc.item(),
                                               b/len(train_loader)*100)
            else:
                s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                s = s.format(loss.item(), acc.item(),
                                          b/len(train_loader)*100)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and b>5: break

        print()
        optimizer.zero_grad()
        mask_train_loss = mask_avg_loss/len(train_loader)
        mask_train_acc = mask_avg_acc/len(train_loader)
        train_avg_loss = avg_loss/len(train_loader)
        train_avg_acc = avg_acc/len(train_loader)
        train_avg_indy = avg_indy_acc/len(train_loader)

        s = "Train - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_avg_acc,
                                                train_avg_indy)
        if hyps['masking_task']:
            stats_string+="Tr. Mask Loss:{:.5f} | Tr. Mask Acc:{:.5f}\n"
            stats_string = stats_string.format(mask_train_loss,
                                               mask_train_acc)
        if c_preds is not None:
            train_c_avg_loss = avg_c_loss/len(train_loader)
            train_c_avg_acc =  avg_c_acc/len(train_loader)
            train_c_avg_indy = avg_c_indy_acc/len(train_loader)
            s = "Train - CLoss:{:.5f} | CAcc:{:.5f} | CIndy:{:.5f}\n"
            stats_string += s.format(train_c_avg_loss,
                                    train_c_avg_acc,
                                    train_c_avg_indy)
        else:
            train_c_avg_loss = 0
            train_c_avg_acc  = 0
            train_c_avg_indy = 0
        if z_preds is not None:
            train_z_avg_loss = avg_z_loss/len(train_loader)
            stats_string+="Train - ZLoss:{:.5f}\n".format(train_z_avg_loss)
        else: train_z_avg_loss = 0
        model.eval()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        avg_c_loss = 0
        avg_c_acc = 0
        avg_c_indy_acc = 0
        avg_z_loss = 0
        mask_avg_loss = 0
        mask_avg_acc = 0
        print("Validating...")
        torch.cuda.empty_cache()
        with torch.no_grad():
            rand_word_batch = int(np.random.randint(0,len(val_loader)))
            for b,tup in enumerate(val_loader):
                x,y = tup[:2]
                targs = y.data[:,1:]
                if hyps['init_decs']:
                    y = train_data.inits.clone().repeat(len(targs),1)
                og_shape = targs.shape
                idx2word = train_data.idx2word
                if hyps['masking_task']:
                    x,y,mask = mask_words(x, y, mask_p=hyps['mask_p'])
                y = y[:,:-1]

                z,c = None,None
                if len(tup) == 3:
                    z = tup[-1].to(DEVICE)
                elif len(tup) == 4: 
                    z,c = tup[-2].to(DEVICE), tup[-1].to(DEVICE)
                    c_targs = c[:,1:]
                    c = c[:,:-1]
                    c_og_shape = c.shape
                    z_targs = z[:,1:]
                    z_og_shape = z_targs.shape
                    z = z[:,:-1]
                preds = model(x.to(DEVICE), y.to(DEVICE), z, c)
                c_preds,z_preds = None, None
                if isinstance(preds,tuple):
                    preds,c_preds,z_preds = preds

                if hyps['masking_task']:
                    # Mask loss and acc
                    targs = targs.reshape(-1)
                    preds = preds.reshape(-1,preds.shape[-1])
                    mask = mask.reshape(-1).bool()
                    idxs = torch.arange(len(mask))[mask]
                    mask_preds = preds[idxs]
                    mask_targs = targs[idxs]
                    mask_loss = lossfxn(mask_preds,mask_targs)
                    mask_avg_loss += mask_loss.item()
                    mask_preds = torch.argmax(mask_preds,dim=-1)
                    mask_acc = (mask_preds==mask_targs).sum().float()
                    mask_acc = mask_acc/mask_preds.numel()
                    mask_avg_acc += mask_acc.item()
                else:
                    mask_acc = torch.zeros(1).to(DEVICE)
                    mask_loss = torch.zeros(1).to(DEVICE)

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
                if c_preds is not None:
                    c_preds = c_preds.reshape(-1,c_preds.shape[-1])
                    c_targs = c_targs.reshape(-1).to(DEVICE)
                    c_bitmask = c_targs!=mask_idx
                    c_loss = lossfxn(c_preds[c_bitmask],
                                     c_targs[c_bitmask])
                    c_preds = torch.argmax(c_preds,dim=-1)
                    sl = c_og_shape[-1]
                    eq = (c_preds==c_targs).float()
                    c_indy_acc = eq.reshape(-1)[c_bitmask].mean()
                    eq[~c_bitmask] = 1
                    eq = eq.reshape(c_og_shape)
                    c_acc = (eq.sum(-1)==sl).float().mean()
                    avg_c_acc += c_acc.item()
                    avg_c_indy_acc += c_indy_acc.item()
                    avg_c_loss += c_loss.item()
                if z_preds is not None:
                    z_targs = z_targs.to(DEVICE)
                    z_bitmask = z_targs > null_loc
                    z_loss = F.mse_loss(z_preds[z_bitmask],
                                        z_targs[z_bitmask])
                    avg_z_loss += z_loss.item()

                if b == rand_word_batch or hyps['exp_name']=="test":
                    rand = int(np.random.randint(0,len(x)))
                    question = x[rand]
                    if len(question.shape) == 1:
                        whr = torch.where(question==mask_idx)[0]
                        endx = len(question)
                        if len(whr)!=0: endx = whr[0].item()
                        question = question[:endx]
                        question=[idx2word[p.item()] for p in question]
                    else: 
                        question = "img based question"
                    targ_samp = targs.reshape(og_shape)[rand]
                    whr = torch.where(targ_samp==mask_idx)[0]
                    endx = len(targ_samp)
                    if len(whr)>0: endx = whr[0].item()
                    targ_samp = targ_samp[:endx]
                    pred_samp = preds.reshape(og_shape)[rand,:endx]
                    idx2word = train_data.idx2word
                    targ_samp = [idx2word[p.item()] for p in targ_samp]
                    pred_samp = [idx2word[p.item()] for p in pred_samp]
                    question = " ".join(question)
                    targ_samp = " ".join(targ_samp)
                    pred_samp = " ".join(pred_samp)
                    if c_preds is not None:
                        ct_samp = c_targs.reshape(c_og_shape)[rand]
                        mask = ct_samp!=mask_idx
                        ct_samp = ct_samp[mask]
                        ct_samp = [idx2word[a.item()] for a in ct_samp]
                        cp_samp=c_preds.reshape(c_og_shape)[rand][mask]
                        cp_samp = [idx2word[p.item()] for p in cp_samp]
                        c_targ_samp = " ".join(ct_samp)
                        c_pred_samp = " ".join(cp_samp)

                    if z_preds is not None:
                        zt_samp = z_targs.reshape(z_og_shape)[rand]
                        mask = zt_samp[:,0] > null_loc

                        zt_samp = zt_samp[mask]
                        s = "{:.3f} "*len(zt_samp[:,0])

                        ztx = s.format(*zt_samp[:,0].tolist())
                        z_targ_sampx = ztx
                        zty = s.format(*zt_samp[:,1].tolist())
                        z_targ_sampy = zty
                        zp_samp = z_preds.reshape(z_og_shape)[rand][mask]
                        zpx = s.format(*zp_samp[:,0].tolist())
                        z_pred_sampx = zpx
                        zpy = s.format(*zp_samp[:,1].tolist())
                        z_pred_sampy = zpy

                if hyps["masking_task"]:
                    s = "Mask Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                    s = s.format(mask_loss.item(), mask_acc.item(),
                                                   b/len(val_loader)*100)
                else:
                    s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
                    s = s.format(loss.item(), acc.item(),
                                              b/len(val_loader)*100)
                print(s, end=len(s)*" " + "\r")
                if hyps['exp_name']=="test" and b > 5: break


        print()
        mask_val_loss = mask_avg_loss/len(val_loader)
        mask_val_acc = mask_avg_acc/  len(val_loader)
        val_avg_loss = avg_loss/len(val_loader)
        val_avg_acc = avg_acc/len(val_loader)
        scheduler.step(val_avg_acc)
        val_avg_indy = avg_indy_acc/len(val_loader)
        stats_string += "Val - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = stats_string.format(val_avg_loss,val_avg_acc,
                                                        val_avg_indy)
        if hyps['masking_task']:
            stats_string+="Val Mask Loss:{:.5f} | Val Mask Acc:{:.5f}\n"
            stats_string=stats_string.format(mask_avg_loss,mask_avg_acc)
        if c_preds is not None:
            val_c_avg_loss = avg_c_loss/len(val_loader)
            val_c_avg_acc = avg_c_acc/len(val_loader)
            val_c_avg_indy = avg_c_indy_acc/len(val_loader)
            s = "Val - CLoss:{:.5f} | CAcc:{:.5f} | CIndy:{:.5f}\n"
            stats_string += s.format(val_c_avg_loss,val_c_avg_acc,
                                                    val_c_avg_indy)
        else:
            val_c_avg_loss = 0
            val_c_avg_acc  = 0
            val_c_avg_indy = 0
        if z_preds is not None:
            val_z_avg_loss = avg_z_loss/len(train_loader)
            stats_string += "Val - ZLoss:{:.5f}\n".format(val_z_avg_loss)
        else: val_z_avg_loss = 0
        stats_string += "Quest: " + question + "\n"
        stats_string += "Targ: " + targ_samp + "\n"
        stats_string += "Pred: " + pred_samp + "\n"
        if c_preds is not None:
            stats_string += "CTarg: " + c_targ_samp + "\n"
            stats_string += "CPred: " + c_pred_samp + "\n"
        if z_preds is not None:
            stats_string += "ZTargX: " + z_targ_sampx + "\n"
            stats_string += "ZPredX: " + z_pred_sampx + "\n"
            stats_string += "ZTargY: " + z_targ_sampy + "\n"
            stats_string += "ZPredY: " + z_pred_sampy + "\n"
        optimizer.zero_grad()

        save_dict = {
            "epoch":epoch,
            "hyps":hyps,
            "train_loss":train_avg_loss,
            "train_acc":train_avg_acc,
            "train_indy":train_avg_indy,
            "mask_train_loss":mask_train_loss,
            "mask_train_acc":mask_train_acc,
            "val_loss":val_avg_loss,
            "val_acc":val_avg_acc,
            "val_indy":val_avg_indy,
            "mask_val_loss":mask_val_loss,
            "mask_val_acc":mask_val_acc,
            "train_c_loss":train_c_avg_loss,
            "train_c_acc":train_c_avg_acc,
            "train_c_indy":train_c_avg_indy,
            "val_c_loss":val_c_avg_loss,
            "val_c_acc":val_c_avg_acc,
            "val_c_indy":val_c_avg_indy,
            "train_z_loss":train_z_avg_loss,
            "val_z_loss":val_z_avg_loss,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
            "word2idx":train_data.word2idx,
            "idx2word":train_data.idx2word,
            "sampled_types":train_data.sampled_types
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

