from scripts.data_preparation import reshape

import copy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from shapely.geometry import Point, Polygon
from tqdm import tqdm
from time import sleep


def initialize_weights(module):
    """Sample inital weights for the convolutional layers from a normal distribution."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.normal_(module.weight, std=0.01)


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train):
    # Set model to training mode
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0

    with tqdm(dataloader, unit="batch") as tepoch:
        # Iterate over data
        for xb, yb in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            xb, yb = xb.to(device), yb.to(device)

            # zero the parameters
            if train:
                optimizer.zero_grad(set_to_none=True)

            # forward
            with torch.set_grad_enabled(train):
                pred = model(xb)
                loss = loss_fn(pred, yb)

                # backward + optimize if in training phase
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()

            # statistics
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader.dataset)
        epoch_rmse = np.sqrt(2 * epoch_loss)
        tepoch.set_postfix(loss=epoch_loss)
        sleep(0.01)
    return epoch_loss, epoch_rmse


def fit(model, loss_fn, scheduler, dataloaders, optimizer, device, writer, NAME, max_epochs, patience):
    best_val_rmse = np.inf
    best_epoch = -1
    best_model_weights = {}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_rmse = run_epoch(model, loss_fn, dataloaders['train'], device, epoch, optimizer, train=True)
        scheduler.step()
        val_loss, val_rmse = run_epoch(model, loss_fn, dataloaders['val'], device, epoch, optimizer=None, train=False)
        print(
            f"Epoch {epoch}/{max_epochs}, train_loss: {train_loss:.3f}, train_rmse: {train_rmse:.3f}, val_loss: {val_loss:.3f}, val_rmse: {val_rmse:.3f}")

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_rmse', train_rmse, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)

        # Save best weights
        if val_rmse < best_val_rmse:
            best_epoch = epoch
            best_val_rmse = val_rmse
            best_model_weights = copy.deepcopy(model.state_dict())

        # Early stopping
        print(
            f"best val_rmse: {best_val_rmse:.3f}, epoch: {epoch}, best_epoch: {best_epoch}, current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    torch.save(best_model_weights, f'{NAME}_best_val_rmse_{np.round(best_val_rmse, 3)}.pth')


def predictTrac(logits, model, E):
    with torch.no_grad():
        S = logits.shape[3]
        mag = S / 104
        conversion = E / (10 * mag)
        return model(logits) * conversion


def errorTrac(inp, target, model, E, plot=False):
    # Calculate error of traction stress field realtive to ground truth as normalized MSE for cell interior only.
    model.eval()
    brdx = np.array(inp['brdx'])  # x-values of predicted cell border
    brdy = np.array(inp['brdy'])  # y-values of predicted cell border
    trac_pred = predictTrac(torch.from_numpy(reshape(np.array(inp['dspl'])[np.newaxis, ])).double(), model, E)  # predict traction field for a single sample
    trac_gt = torch.from_numpy(reshape(np.array(target['trac'])[np.newaxis, ])).double()

    zipped = np.array(list(zip(brdx[0], brdy[0])))  # array with (x,y) pairs of cell border coordinates
    polygon = Polygon(zipped)  # create polygon

    interior = np.zeros((inp['dspl'].shape[0], inp['dspl'].shape[1]), dtype=int)  # create all zero matrix
    for i in range(len(interior)):  # set all elements in interior matrix to 1 that actually lie within the cell
        for j in range(len(interior[i])):
            point = Point(i, j)
            if polygon.contains(point):
                interior[i][j] = 1
    if plot:
        p = gpd.GeoSeries(polygon)
        p.plot()
        plt.show()

    # update prediction and ground truth by discarding areas outside of cell borders
    interior = torch.from_numpy(interior)
    trac_pred[-1, -1, 0, :, :] = trac_pred[-1, -1, 0, :, :] * interior
    trac_pred[-1, -1, 1, :, :] = trac_pred[-1, -1, 1, :, :] * interior
    trac_gt[-1, -1, 0, :, :] = trac_gt[-1, -1, 0, :, :] * interior
    trac_gt[-1, -1, 1, :, :] = trac_gt[-1, -1, 1, :, :] * interior

    # compute rmse
    normalization = torch.count_nonzero(interior * torch.ones(size=interior.shape))
    mse = torch.sum(((trac_pred[-1, -1, 0, :, :] - trac_gt[-1, -1, 0, :, :]) ** 2 + (
                trac_pred[-1, -1, 1, :, :] - trac_gt[-1, -1, 1, :, :]) ** 2) / normalization)
    rmse = torch.sqrt(mse)
    msm = torch.sum((trac_pred[-1, -1, 0, :, :] ** 2 + trac_gt[-1, -1, 1, :, :] ** 2) / normalization)
    rmsm = np.sqrt(msm)
    error = rmse / rmsm

    return error


def test(inputs, targets, model, E, plot=False):
    losses = {}
    # Iterate over data
    for i in range(len(inputs)):
        loss = errorTrac(inputs[i], targets[i], model, E, plot)
        losses[inputs[i]['name']] = loss

    return losses

