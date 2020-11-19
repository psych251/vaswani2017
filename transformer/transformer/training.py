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
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from nltk.translate.bleu_score import corpus_bleu


MASK  = "<MASK>"
START = "<START>"
STOP  = "<STOP>"

def train(gpu, hyps, verbose=True):
    """
    gpu: int
        the gpu for this training process
    hyps: dict
        contains all relavent hyperparameters
    """
    rank = 0
    if hyps['multi_gpu']:
        rank = hyps['n_gpus']*hyps['node_rank'] + gpu
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=hyps['world_size'],
            rank=rank)
    verbose = verbose and rank==0
    hyps['rank'] = rank

    torch.cuda.set_device(gpu)
    test_batch_size = try_key(hyps,"test_batch_size",False)
    if test_batch_size and verbose:
        print("Testing batch size!! No saving will occur!")
    hyps['main_path'] = try_key(hyps,'main_path',"./")
    if "ignore_keys" not in hyps:
        hyps['ignore_keys'] = ["n_epochs", "batch_size",
                                "max_context","rank", "n_loss_loops"]
    checkpt,hyps = get_resume_checkpt(hyps)
    if checkpt is None and rank==0:
        hyps['exp_num']=get_exp_num(hyps['main_path'], hyps['exp_name'])
        hyps['save_folder'] = get_save_folder(hyps)
    if rank>0: hyps['save_folder'] = "placeholder"
    if not os.path.exists(hyps['save_folder']) and\
                    not test_batch_size and rank==0:
        os.mkdir(hyps['save_folder'])
    # Set manual seed
    hyps['seed'] = try_key(hyps, 'seed', int(time.time()))+rank
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])
    hyps['MASK'] = MASK
    hyps['START'] = START
    hyps['STOP'] = STOP

    model_class = hyps['model_class']
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)
    if not hyps['init_decs'] and not hyps['ordered_preds'] and verbose:
        s = "WARNING!! You probably want to set ordered preds to True "
        s += "with your current configuration!!"
        print(s)

    if verbose:
        print("Retreiving Dataset")
    if "shuffle_split" not in hyps and hyps['shuffle']:
        hyps['shuffle_split'] = True
    train_data,val_data = datas.get_data(hyps)

    hyps['enc_slen'] = train_data.X.shape[-1]
    hyps['dec_slen'] = train_data.Y.shape[-1]
    hyps["mask_idx"] = train_data.X_tokenizer.token_to_id(MASK)
    hyps["dec_mask_idx"] = train_data.Y_tokenizer.token_to_id(MASK)
    hyps['n_vocab'] = train_data.X_tokenizer.get_vocab_size()
    hyps['n_vocab_out'] = train_data.Y_tokenizer.get_vocab_size()

    train_loader = datas.VariableLengthSeqLoader(train_data,
                                    samples_per_epoch=1000,
                                    shuffle=hyps['shuffle'])
    val_loader = datas.VariableLengthSeqLoader(val_data,
                                    samples_per_epoch=50,
                                    shuffle=True)

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)
    model.cuda(gpu)
    lossfxn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    if hyps['multi_gpu']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O0')
        model = DDP(model)
    # Load State Dicts if Resuming Training
    if checkpt is not None:
        if verbose:
            print("Loading state dicts from", hyps['save_folder'])
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
        epoch = checkpt['epoch']
        if hyps['multi_gpu'] and "amp_dict" in checkpt:
            amp.load_state_dict(checkpt['amp_dict'])
    else:
        epoch = -1
    scheduler = custmods.VaswaniScheduler(optimizer, hyps['emb_size'])
    if verbose:
        print("Beginning training for {}".format(hyps['save_folder']))
        print("train shape:", (len(train_data),*train_data.X.shape[1:]))
        print("val shape:", (len(val_data),*val_data.X.shape[1:]))
    if not test_batch_size:
        record_session(hyps,model)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    mask_idx = train_data.Y_mask_idx
    step_num = 0 if checkpt is None else try_key(checkpt,'step_num',0)
    checkpt_steps = 0
    if verbose: print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        if verbose:
            print("Epoch:{} | Step: {} | Model:{}".format(epoch,
                                                 step_num,
                                                 hyps['save_folder']))
            print("Training...")
        starttime = time.time()
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        checkpt_loss = 0
        checkpt_acc = 0
        model.train()
        optimizer.zero_grad()
        for b,(x,y) in enumerate(train_loader):
            if test_batch_size:
                x,y = train_loader.get_largest_batch(b)
            torch.cuda.empty_cache()
            targs = y.data[:,1:]
            og_shape = targs.shape
            y = y[:,:-1]
            logits = model(x.cuda(non_blocking=True),
                           y.cuda(non_blocking=True))
            preds = torch.argmax(logits,dim=-1)

            if epoch % 3 == 0 and b == 0 and verbose:
                inp = x.data[0].cpu().numpy()
                trg = targs.data[0].numpy()
                prd = preds.data[0].cpu().numpy()
                print("Inp:", train_data.X_idxs2tokens(inp))
                print("Targ:", train_data.Y_idxs2tokens(trg))
                print("Pred:", train_data.Y_idxs2tokens(prd))

            # Tot loss
            logits = logits.reshape(-1,logits.shape[-1])
            targs = targs.reshape(-1).cuda(non_blocking=True)
            bitmask = targs!=mask_idx
            loss = lossfxn(logits[bitmask],targs[bitmask])

            loss = loss/hyps['n_loss_loops']
            if hyps['multi_gpu']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if b % hyps['n_loss_loops'] == 0 or b == len(train_loader)-1:
                optimizer.step()
                optimizer.zero_grad()
                step_num += 1
                scheduler.update_lr(step_num)

            with torch.no_grad():
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
                checkpt_acc += acc.item()
                checkpt_loss += loss.item()

            s = "Loss:{:.5f} | Acc:{:.5f} | {:.0f}%"
            s = s.format(loss.item(), acc.item(),
                                      b/len(train_loader)*100)
            if verbose: print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and b>5: break

        optimizer.zero_grad()
        train_avg_loss = avg_loss/len(train_loader)
        train_avg_acc = avg_acc/len(train_loader)
        train_avg_indy = avg_indy_acc/len(train_loader)

        s = "Ending Step Count: {}\n".format(step_num)
        s = s+"Train - Loss:{:.5f} | Acc:{:.5f} | Indy:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_avg_acc,
                                                train_avg_indy)

        ###### VALIDATION
        model.eval()
        avg_bleu = 0
        avg_loss = 0
        avg_acc = 0
        avg_indy_acc = 0
        if verbose: print("\nValidating...")
        torch.cuda.empty_cache()
        if rank==0:
            with torch.no_grad():
                rand_word_batch = int(np.random.randint(0,
                                         len(val_loader)))
                for b,(x,y) in enumerate(val_loader):
                    if test_batch_size:
                        x,y = val_loader.get_largest_batch(b)
                    targs = y.data[:,1:]
                    og_shape = targs.shape
                    y = y[:,:-1]
                    preds = model(x.cuda(non_blocking=True),
                                  y.cuda(non_blocking=True))

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
                    bleu_prds=preds.reshape(og_shape).data.cpu().numpy()
                    bleu = corpus_bleu(bleu_trgs[:,None,:],bleu_prds)
                    avg_bleu += bleu
                    avg_acc += acc.item()
                    avg_indy_acc += indy_acc.item()
                    avg_loss += loss.item()

                    if b == rand_word_batch or hyps['exp_name']=="test":
                        rand = int(np.random.randint(0,len(x)))
                        inp = x.data[rand].cpu().numpy()
                        inp_samp = val_data.X_idxs2tokens(inp)
                        trg = targs.reshape(og_shape)[rand].data.cpu()
                        targ_samp = val_data.Y_idxs2tokens(trg.numpy())
                        prd = preds.reshape(og_shape)[rand].data.cpu()
                        pred_samp = val_data.Y_idxs2tokens(prd.numpy())
                    s="Loss:{:.5f} | Acc:{:.5f} | Bleu:{:.5f} | {:.0f}%"
                    s = s.format(loss.item(), acc.item(), bleu,
                                              b/len(val_loader)*100)
                    if verbose: print(s, end=len(s)*" " + "\r")
                    if hyps['exp_name']=="test" and b > 5: break

            if verbose: print()
            val_avg_bleu = avg_bleu/len(val_loader)
            val_avg_loss = avg_loss/len(val_loader)
            val_avg_acc = avg_acc/len(val_loader)
            val_avg_indy = avg_indy_acc/len(val_loader)
            stats_string += "Val- Loss:{:.5f} | Acc:{:.5f} | "
            stats_string += "Indy:{:.5f}\nVal Bleu: {:.5f}\n"
            stats_string = stats_string.format(val_avg_loss,val_avg_acc,
                                                            val_avg_indy,
                                                            val_avg_bleu)
            stats_string += "Inp: "  + inp_samp  + "\n"
            stats_string += "Targ: " + targ_samp + "\n"
            stats_string += "Pred: " + pred_samp + "\n"
        optimizer.zero_grad()

        if not test_batch_size and rank==0:
            save_dict = {
                "epoch":epoch,
                "step_num":step_num,
                "hyps":hyps,
                "train_loss":train_avg_loss,
                "train_acc":train_avg_acc,
                "train_indy":train_avg_indy,
                "val_bleu":val_avg_bleu,
                "val_loss":val_avg_loss,
                "val_acc":val_avg_acc,
                "val_indy":val_avg_indy,
                "state_dict":model.state_dict(),
                "optim_dict":optimizer.state_dict(),
            }
            if hyps['multi_gpu']: save_dict['amp_dict']=amp.state_dict()
            save_name = "checkpt"
            save_name = os.path.join(hyps['save_folder'],save_name)
            io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                del_prev_sd=hyps['del_prev_sd'])
        stats_string += "Exec time: {}\n".format(time.time()-starttime)
        if verbose: print(stats_string)
        s = "Epoch:{} | Model:{}\n".format(epoch, hyps['save_folder'])
        stats_string = s + stats_string
        log_file = os.path.join(hyps['save_folder'],
                                "training_log"+str(rank)+".txt")
        if not test_batch_size:
            with open(log_file,'a') as f:
                f.write(str(stats_string)+'\n')
    if rank==0:
        del save_dict['state_dict']
        del save_dict['optim_dict']
        del save_dict['hyps']
        save_dict['save_folder'] = hyps['save_folder']
        return save_dict
    return None

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

