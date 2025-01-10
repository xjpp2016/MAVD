import math
import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype = np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)
    
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("iteration: {}\n".format(test_info["iteration"][-1]))
    fo.write("m_ap: {:.4f}\n".format(test_info["m_ap"][-1]))


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat

def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]


def uniform_extract(feat, t_max):
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def process_test_feat(feat, length):
    tem_len = len(feat)
    num = math.ceil(tem_len / length)
    if len(feat) < length:
        return pad(feat, length)
    else:
        return pad(feat, num * length)


