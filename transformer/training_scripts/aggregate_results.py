"""
This script serves as a way to create csv files with the results of
the training. These csv files can then be used with pandas
"""
import locgame.save_io as locio
import ml_utils.save_io as mlio
import ml_utils.analysis as mlanl
from ml_utils.utils import try_key
import pandas as pd
import os
import sys

if __name__ == "__main__":
    argued_folders = sys.argv[1:]
    model_folders = []
    for folder in argued_folders:
        if not mlio.is_model_folder(folder):
            model_folders += mlio.get_model_folders(folder,True)
        else:
            model_folders += [folder]
    print("Model Folders:", model_folders)
    for model_folder in model_folders:
        checkpts = mlio.get_checkpoints(model_folder)
        if len(checkpts) == 0: continue
        table = mlanl.get_table(mlio.load_checkpoint(checkpts[0]))
        for checkpt in checkpts:
            chkpt = mlio.load_checkpoint(checkpt)
            for k in table.keys():
                if k in set(chkpt.keys()):
                    table[k].append(chkpt[k])
        df = pd.DataFrame(table)
        df['seed'] = try_key(chkpt['hyps'],'seed',-1)
        save_path = os.path.join(model_folder, "model_data.csv")
        df.to_csv(save_path, sep="!", index=False, header=True)
