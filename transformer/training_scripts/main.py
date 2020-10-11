import os
import sys
import crab as crab
import torch
import numpy as np
import shutil
import time
import numpy as np
import select

if torch.cuda.is_available:
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    hyps = crab.utils.load_json(sys.argv[1])
    print()
    print("Using hyperparams file:", sys.argv[1])
    if len(sys.argv) < 3:
        ranges = {"lr": [hyps['lr']]}
    else:
        ranges = crab.utils.load_json(sys.argv[2])
        print("Using hyperranges file:", sys.argv[2])
    print()

    hyps_str = ""
    for k,v in hyps.items():
        if k not in ranges:
            hyps_str += "{}: {}\n".format(k,v)
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k,v) for k,v in ranges.items()]))

    # Random Seeds
    seed = 3
    if "rand_seed" in hyps:
        seed = hyps['rand_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    main_path = hyps['exp_name']
    if "save_root" in hyps:
        hyps['save_root'] = os.path.expanduser(hyps['save_root'])
        if not os.path.exists(hyps['save_root']):
            os.mkdir(hyps['save_root'])
        main_path = os.path.join(hyps['save_root'], main_path)
    sleep_time = 8
    if os.path.exists(main_path):
        _, subds, _ = next(os.walk(main_path))
        dirs = []
        for d in subds:
            splt = d.split("_")
            if len(splt) >= 2 and splt[0] == hyps['exp_name']:
                dirs.append(d)
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        if len(dirs) > 0:
            s = "Overwrite last folder {}? (No/yes)".format(dirs[-1])
            print(s)
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
            if i and "y" in sys.stdin.readline().strip().lower():
                path = os.path.join(main_path, dirs[-1])
                shutil.rmtree(path, ignore_errors=True)
        else:
            s = "You have {} seconds to cancel experiment name {}:"
            print(s.format(sleep_time, hyps['exp_name']))
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    else:
        s = "You have {} seconds to cancel experiment name {}:"
        print(s.format(sleep_time, hyps['exp_name']))
        i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    print()

    keys = list(ranges.keys())
    start_time = time.time()
    crab.training.hyper_search(hyps,ranges)
    print("Total Execution Time:", time.time() - start_time)

