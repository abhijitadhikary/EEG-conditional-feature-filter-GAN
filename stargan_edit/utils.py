import os
import scipy.io as sio

def load_mat(filename):
    return sio.loadmat(filename)

def convert(input, min_value=-1, max_value=1):
    min_source = input.min()
    max_source = input.max()

    a = (max_value - min_value) / (max_source - min_source)
    b = max_value - a * max_source
    output = (a * input + b)
    return output

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)