import torch
import pickle
from crab.models import *
import crab.utils as utils
import os

def save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                       del_prev_sd=True):
    """
    Saves a dictionary that contains a statedict
    
    save_dict: dict
        a dictionary containing all the things you want to save
    save_name: str
        the path to save the dict to.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    """
    if del_prev_sd:
        prev_path = "{}_{}{}".format(save_name,epoch-1,ext)
        prev_path = os.path.abspath(os.path.expanduser(prev_path))
        if os.path.exists(prev_path):
            device = torch.device("cpu")
            data = torch.load(prev_path, map_location=device)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key or "optim_dic" in key:
                    del data[key]
            torch.save(data, prev_path)
        elif save_dict['epoch'] != 0:
            print("Failed to find previous checkpoint", prev_path)
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder.

    folder: str
        path to the folder of interest

    Returns:
    checkpts: list of str
        the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    print(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    sort_key = lambda x: int(x.split(".")[-2].split("_")[-1])
    checkpts = sorted(checkpts, key=sort_key)
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    x: str
    """
    splt = x.split("/")[-1].split("_")
    for i,s in enumerate(splt[1:]):
        try:
            return int(s)
        except:
            pass
    assert False

def get_model_folders(main_folder):
    """
    Returns a list of paths to the model folders contained within the
    argued main_folder

    main_folder - str
        path to main folder

    Returns:
        list of folders without full extension
    """
    folders = []
    main_folder = os.path.expanduser(main_folder)
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content or "hyperparams.txt" == content:
                    folders.append(sub_d)
                    break
    return sorted(folders, key=foldersort)

def load_checkpoint(path):
    """
    Can load a specific model file both architecture and state_dict
    if the file contains a model_state_dict key, or can just load the
    architecture.

    path: str
        path to checkpoint file
    """
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        hyps = get_hyps(path)
        path = checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    return data

def load_model(path, load_sd=True, verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        hyps = get_hyps(path)
        path = checkpts[-1]
    data = load_checkpoint(path)
    if 'hyps' in data:
        kwargs = data['hyps']
    elif 'model_hyps' in data:
        kwargs = data['model_hyps']
    elif hyps is not None:
        kwargs = hyps
    else:
        assert False, "Cannot find architecture arguments"
    model = globals()[kwargs['model_class']](**kwargs)
    if "state_dict" in data and load_sd:
        try:
            model.load_state_dict(data['state_dict'])
        except:
            sd = data['state_dict']
            keys = list(sd.keys())
            for k in keys:
                splt = k.split(".")
                if "transformer" == splt[0]:
                    new_key = ["seqmodel"] + [x for x in splt[1:]]
                    new_key = ".".join(new_key)
                    sd[new_key] = sd[k]
                    del sd[k]
                    if verbose:
                        print("renaming {} to {}".format(k,new_key))
                elif "module" == splt[0]:
                    new_key = [x for x in splt[1:]]
                    new_key = ".".join(new_key)
                    sd[new_key] = sd[k]
                    del sd[k]
                    if verbose:
                        print("renaming {} to {}".format(k,new_key))
            model.load_state_dict(sd)
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    hyps_json = os.path.join(folder, "hyperparams.json")
    hyps = utils.load_json(hyps_json)
    return hyps

def get_next_exp_num(exp_name):
    """
    Finds the next open experiment id number.

    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    exp_nums = set()
    for folder in folders:
        exp_num = foldersort(folder)
        exp_nums.add(exp_num)
    for i in range(len(exp_nums)):
        if i not in exp_nums:
            return i
    return len(exp_nums)

def exp_num_exists(exp_num, exp_name):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_name: str
        path to the main experiment folder that contains the model
        folders
    """
    folders = get_model_folders(exp_name)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def make_save_folder(hyps):
    """
    Creates the save name for the model.

    hyps: dict
        keys:
            exp_name: str
            exp_num: int
            search_keys: str
    """
    save_folder = "{}/{}_{}".format(hyps['exp_name'],
                                    hyps['exp_name'],
                                    hyps['exp_num'])
    save_folder += hyps['search_keys']
    return save_folder

