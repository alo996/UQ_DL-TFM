import warnings

from scipy.io import loadmat

warnings.filterwarnings('ignore')
import copy
import datetime
import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import tqdm
from MultiTask import append_predictions_and_targets, compute_mse_for_noise_levels, compute_dtma_for_noise_levels
sys.path.append('../ViT architecture/Architecture tryouts/DPT')
from VisionTransformer_working_for_DPT import VisionTransformer

# Setup
random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
cudnn.benchmark = True

torch.cuda.empty_cache()
torch.set_printoptions(precision=6)
device = torch.device('cpu')
print(f"Running on device: {device}")

# Load models
vit = VisionTransformer(dspl_size=104,
                        patch_size=8,
                        embed_dim=128,
                        depth=4,
                        n_heads=4,
                        mlp_ratio=1.,
                        p=0.05,
                        attn_p=0.05,
                        drop_path=0.).float()
path_to_vit_new = '/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/ViT architecture/Architecture tryouts/logs_and_weights/ViT-clean_2023-Feb-26 22:41:48/ViT-clean_2023-Feb-26 22:41:48.pth'
if torch.cuda.is_available():
    vit.load_state_dict(torch.load(path_to_vit_new)['best_model_weights'], strict=True)
else:
    vit.load_state_dict(torch.load(path_to_vit_new, map_location=torch.device('cpu'))['best_model_weights'], strict=True)

cnn = keras.models.load_model('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm/CNN_noisy-2023-Mar-14 12:17:31_checkpoint.h5')

# Load data
dspl_test = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Test data/resolution_104/allDisplacements.h5', 'r')['dspl'])
trac_test = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Test data/resolution_104/allTractions.h5', 'r')['trac'])
dspl_test = np.concatenate([dspl_test[i] for i in range(dspl_test.shape[0])], axis=0, dtype=np.float32)
trac_test = np.concatenate([trac_test[i] for i in range(trac_test.shape[0])], axis=0, dtype=np.float32)
dspl_test = dspl_test[:100]
Y_test = torch.from_numpy(np.moveaxis(trac_test[:100], 3, 1))

std_dspl = np.std(dspl_test, axis=(1,2,3))
test_sets = {}
noise_sets = {}
for i in range(1, 11):
    test_set = np.zeros((dspl_test.shape))
    noise_set = np.zeros((dspl_test.shape))
    for j in range(len(dspl_test)):
        sigma = (i/100) * std_dspl[j]
        cov = [[sigma**2,0],[0,sigma**2]]
        noise = np.random.multivariate_normal(np.array([0,0]), cov, (104, 104))
        test_set[j] = dspl_test[j] + noise
        noise_set[j] = noise
    test_sets[f'{i}'] = np.moveaxis(test_set, 3, 1)
    noise_sets[f'{i}'] = np.moveaxis(noise_set, 3, 1)

# Predictions
vit.eval()
vit_predictions = {}
cnn_predictions = {}
for num, test_set in test_sets.items():
    vit_predictions[num] = vit(torch.tensor(test_set).float())
    cnn_predictions[num] = cnn.predict(np.moveaxis(np.array(test_set), 1, 3))

# Load bfftc data and reshape/reorder
bfftc_prediction_sets = {}
bfftc_displacement_sets = {}

directory = "/home/alexrichard/LRZ Sync+Share/ML in Physics/Repos/Easy-to-use_TFM_package-master/test_data/Artificial patch data/Predictions"

for i, _ in enumerate(os.listdir(directory), 1):
    bfftc_prediction = np.zeros((Y_test.shape[0], 2, 102, 102))
    bfftc_displacement = np.zeros((Y_test.shape[0], 2, 102, 102))
    for j, file in enumerate(os.listdir(f'{directory}/{i}')):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"):
            bfft_pred = loadmat(f'{directory}/{i}/{filename}')['TFM_results']['traction'][0][0].T.reshape((2, 102, 102),
                                                                                                          order='F')
            bfft_dspl = loadmat(f'{directory}/{i}/{filename}')['TFM_results']['displacement'][0][0].T.reshape(
                (2, 102, 102), order='F')
            bfftc_prediction[j] = bfft_pred
            bfftc_displacement[j] = bfft_dspl
    bfftc_prediction_sets[f'{i}'] = bfftc_prediction
    bfftc_displacement_sets[f'{i}'] = bfftc_displacement

bfftc_prediction_sets_trimmed = {}
ground_truth_sets_trimmed = {}
noisy_X_test_sets_trimmed = {}

for num, pred in tqdm.tqdm(bfftc_prediction_sets.items()):
    bfftc_prediction_set_trimmed = torch.zeros((test_sets[num].shape[0], test_sets[num].shape[1], 98, 98))
    ground_truths_trimmed = torch.zeros((test_sets[num].shape[0], 3, 98, 98))
    X_test_noisy = torch.zeros(test_sets[num].shape)
    for i, sample in enumerate(bfftc_displacement_sets[num]):
        for j, dspl in enumerate(torch.tensor(test_sets[num])):
            if torch.allclose(dspl[:, 1:103, 1:103].float(), torch.tensor(sample).float(), atol=1e-02, rtol=1):
                # print(f'Set {num}: sample {i} matches with dspl {j}')
                bfftc_prediction_set_trimmed[i] = torch.tensor(bfftc_prediction_sets[num][i, :, 3:101, 3:101]).float()
                ground_truths_trimmed[i] = Y_test[j, :, 3:101, 3:101].float()
                X_test_noisy[i] = dspl.float()
    bfftc_prediction_sets_trimmed[num] = bfftc_prediction_set_trimmed
    ground_truth_sets_trimmed[num] = ground_truths_trimmed
    noisy_X_test_sets_trimmed[num] = X_test_noisy


# Predictions
vit.eval()
vit_predictions = {}
cnn_predictions = {}
for num, test_set in test_sets.items():
    vit_predictions[num] = vit(torch.tensor(test_set).float())
    cnn_predictions[num] = cnn.predict(np.moveaxis(np.array(test_set), 1, 3))

vit_metrics = {}
cnn_metrics = {}
bfftc_metrics = {}


# MSE
vit_metrics['mse'], cnn_metrics['mse'], bfftc_metrics['mse'] = compute_mse_for_noise_levels(vit_predictions, cnn_predictions, bfftc_prediction_sets_trimmed, ground_truth_sets_trimmed, noisy_X_test_sets_trimmed)
print(vit_metrics['mse'])
print(cnn_metrics['mse'])
print(bfftc_metrics['mse'])

appended_vit_predictions = {}
appended_vit_targets = {}
appended_cnn_predictions = {}
appended_cnn_targets = {}
appended_bfftc_predictions = {}
appended_bfftc_targets = {}

noise_levels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
for noise_level in noise_levels:
    appended_vit_predictions[noise_level], appended_vit_targets[noise_level] = append_predictions_and_targets(vit_predictions[noise_level][:, :, 3:101, 3:101], ground_truth_sets_trimmed[noise_level], device)
    appended_cnn_predictions[noise_level], appended_cnn_targets[noise_level] = append_predictions_and_targets(torch.tensor(np.moveaxis(cnn_predictions[noise_level], 3, 1))[:, :, 3:101, 3:101],ground_truth_sets_trimmed[noise_level], device)
    appended_bfftc_predictions[noise_level], appended_bfftc_targets[noise_level] = append_predictions_and_targets(bfftc_prediction_sets_trimmed[noise_level], ground_truth_sets_trimmed[noise_level], device)


vit_metrics['dtma'], cnn_metrics['dtma'], bfftc_metrics['dtma'] = compute_dtma_for_noise_levels(appended_vit_predictions, appended_vit_targets, appended_cnn_predictions, appended_cnn_targets, appended_bfftc_predictions, appended_bfftc_targets, device)
print(vit_metrics['mse'])
print(cnn_metrics['dtma'])
print(bfftc_metrics['dtma'])