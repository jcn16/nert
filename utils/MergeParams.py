import numpy as np
import os
import json
import random

root='/media/jcn/新加卷/Nerf/Datasets/Twindom_512'

param_path=os.path.join(root,'param')
child_dirs=os.listdir(param_path)

all={}

for dir in child_dirs:
    present_path=os.path.join(param_path,dir)
    save_dir=os.path.join(root,'json',dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(360):
        extric_path=os.path.join(present_path,f'{i}_extrinsic.npy')
        intric_path=os.path.join(present_path,f'{i}_intrinsic.npy')
        extric=np.load(extric_path)
        intric=np.load(intric_path)
        all[str(i)]={'extric':extric.tolist(),'intric':intric.tolist()}

    json_str = json.dumps(all, indent=4)
    with open(os.path.join(save_dir, 'all.json'), 'w') as json_file:
        json_file.write(json_str)

    all_idxs=list(range(360))
    random.shuffle(all_idxs)

    train={}
    for i in range(180):
        extric_path=os.path.join(present_path,f'{all_idxs[i]}_extrinsic.npy')
        intric_path=os.path.join(present_path,f'{all_idxs[i]}_intrinsic.npy')
        extric=np.load(extric_path)
        intric=np.load(intric_path)
        train[str(all_idxs[i])]={'extric':extric.tolist(),'intric':intric.tolist()}

    json_str = json.dumps(train, indent=4)
    with open(os.path.join(save_dir, 'train.json'), 'w') as json_file:
        json_file.write(json_str)

    val = {}
    for i in range(180,360):
        extric_path = os.path.join(present_path, f'{all_idxs[i]}_extrinsic.npy')
        intric_path = os.path.join(present_path, f'{all_idxs[i]}_intrinsic.npy')
        extric = np.load(extric_path)
        intric = np.load(intric_path)
        val[str(all_idxs[i])] = {'extric': extric.tolist(), 'intric': intric.tolist()}

    json_str = json.dumps(val, indent=4)
    with open(os.path.join(save_dir, 'val.json'), 'w') as json_file:
        json_file.write(json_str)