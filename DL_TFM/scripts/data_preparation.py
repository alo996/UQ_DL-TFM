import numpy as np
import os
from scipy.io import loadmat


def matFiles_to_npArray(path):
    instances = []
    for i, filename in enumerate(os.listdir(path)):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            instance = loadmat(f)
            if '__header__' in instance: del instance['__header__']
            if '__version__' in instance: del instance['__version__']
            if '__globals__' in instance: del instance['__globals__']
            instance['name'] = filename
            instances = np.append(instances, instance)
        else:
            continue

    return np.array(instances)


def reshape(array):
    """current shape is (samples, width, height, depth), reshape to (samples, channels, depth, height, width)"""
    return np.moveaxis(array[:, np.newaxis], [2, 3, 4], [-1, 3, 2])


def extract_fields(dataset):
    if 'dspl' in dataset[0]:
        return np.array([item['dspl'] for item in dataset])
    else:
        return np.array([item['trac'] for item in dataset])
