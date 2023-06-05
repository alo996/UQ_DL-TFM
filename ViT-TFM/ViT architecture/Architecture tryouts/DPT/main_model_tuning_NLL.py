from time import sleep
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import argparse
from datetime import datetime
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ray
import os
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import VisionTransformer_working_for_DPT as vit

def load_data():
    dspl_train_2 = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/allDisplacements.h5', 'r')['dspl'], dtype="float32")
    dspl_train_2_full = np.moveaxis(np.concatenate([dspl_train_2[i] for i in range(dspl_train_2.shape[0])], axis=0), 3, 1)
    del dspl_train_2

    sigma_bar = 0.005 * np.mean(np.var(dspl_train_2_full, axis=(1, 2, 3)))
    print(f'percentage for noise level: {0.005}')
    print(f'sigma_bar: {sigma_bar}')
    cov = [[sigma_bar, 0], [0, sigma_bar]]
    for i, x in tqdm(enumerate(dspl_train_2_full), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl_train_2_full[i] = x + noise
    X_train = torch.from_numpy(dspl_train_2_full).float()
    del dspl_train_2_full
    print(f'X_train_full.shape is {X_train.shape}')

    trac_train_2 = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/allTractions.h5', 'r')['trac'], dtype="float32")
    trac_train_2_full = np.moveaxis(np.concatenate([trac_train_2[i] for i in range(trac_train_2.shape[0])], axis=0), 3, 1)
    del trac_train_2
    Y_train = torch.from_numpy(trac_train_2_full).float()
    del trac_train_2_full
    train_dataset = TensorDataset(X_train.float(), Y_train)
    del X_train, Y_train
    print("Train set ready")

    # Val
    dspl_val = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/allDisplacements.h5', 'r')['dspl'], dtype="float32")
    trac_val = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/allTractions.h5', 'r')['trac'], dtype="float32")
    dspl_val_full = np.moveaxis(np.concatenate([dspl_val[i] for i in range(dspl_val.shape[0])], axis=0), 3, 1)
    del dspl_val
    trac_val_full = np.moveaxis(np.concatenate([trac_val[i] for i in range(trac_val.shape[0])], axis=0), 3, 1)
    del trac_val

    sigma_bar_val = 0.005 * np.mean(np.var(dspl_val_full, axis=(1, 2, 3)))
    print(f'percentage for noise level: {0.005}')
    print(f'sigma_bar_val: {sigma_bar_val}')
    cov = [[sigma_bar_val, 0], [0, sigma_bar_val]]
    for i, x in tqdm(enumerate(dspl_val_full), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl_val_full[i] = x + noise
    X_val = torch.from_numpy(dspl_val_full).float()
    del dspl_val_full
    Y_val = torch.from_numpy(trac_val_full).float()
    del trac_val_full
    val_dataset = TensorDataset(X_val.float(), Y_val)
    del X_val, Y_val

    return train_dataset, val_dataset


def leggo(config):
    # Set seeds
    torch.manual_seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Create model
    print('model is prepared')
    vit_model = vit.VisionTransformer3(dspl_size=104,
                                      patch_size=8,
                                      embed_dim=128,
                                      depth=4,
                                      n_heads=4,
                                      mlp_ratio=1.0,
                                      p=0.05,
                                      attn_p=0.05,
                                      drop_path=0.).float()


    n_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"Number of model parameters to optimize: {n_params}")

    # Specify device
    device = torch.device('cuda')
    print(f"Running on device: {device}")

    # Loss and optimizer
    loss = torch.nn.GaussianNLLLoss()
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=0.001, weight_decay=0.0005, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

    trainset, testset = load_data()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=8,
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        testset,
        batch_size=8,
        shuffle=True,
        num_workers=8)

    dataloaders = {'train': trainloader, 'val': valloader}
    '''
    if args.continue_training:
        NAME = 'ViT-clean_2023-Feb-26 22:41:48'
        writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
        checkpoint = torch.load(f'logs_and_weights/{NAME}/{NAME}.pth')
        vit_model.load_state_dict(checkpoint['final_model_weights'], strict=True)
        vit_model.to(device)
        optimizer.load_state_dict(checkpoint['final_optimizer_state_dict'])
        event_acc = EventAccumulator(f'logs_and_weights/{NAME}')
        event_acc.Reload()
        global_step = event_acc.Scalars(tag='train_loss')[-1].step
    '''

    NAME = "ViT-NLL_{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
    global_step = 0
    checkpoint = None

    fit(vit_model,
        loss,
        dataloaders,
        optimizer,
        device,
        writer,
        NAME,
        100,
        10,
        global_step,
        checkpoint=checkpoint,
        scheduler=scheduler,
        beta=config['beta'])

    writer.flush()
    writer.close()


def main(num_samples, max_num_epochs, gpus_per_trial):
    config = {
        "beta": tune.quniform(0.0, 1.0, 0.05)
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=3)

    print("scheduler set up")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(leggo),
            resources={"cpu": 20, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="val_NLL",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )

    print("tuner set up")

    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["val_loss"]))



def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience, global_step, checkpoint=None, scheduler=None, beta=0.5):
    print(f'beta used is: {beta}')
    if checkpoint is not None:
        best_val_nll = checkpoint['best_val_nll']
        best_epoch = checkpoint['best_epoch']
        best_model_weights = checkpoint['best_model_weights']
    else:
        best_val_nll = np.inf
        best_epoch = 0
        best_model_weights = None

    model.to(device)
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

    for epoch in range(1, max_epochs + 1):
        train_nll, train_mse = run_epoch(model, loss_fn, dataloaders['train'], device, epoch + global_step, optimizer, train=True, beta=beta)
        val_nll, val_mse = run_epoch(model, loss_fn, dataloaders['val'], device, epoch + global_step, optimizer=None, train=False, beta=beta)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar('train_NLL', train_nll, epoch + global_step)
        writer.add_scalar('train_MSE', train_mse, epoch + global_step)
        writer.add_scalar('val_NLL', val_nll, epoch + global_step)
        writer.add_scalar('val_MSE', val_mse, epoch + global_step)

        # Save best weights.
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch + global_step
            best_model_weights = model.state_dict()

        print(
            '\n'
            f"Epoch: {epoch + global_step}/{max_epochs + global_step}" '\n'
            f"train nll: {np.round(train_nll, 12)}" '\n'
            f"train mse: {np.round(train_mse, 12)}" '\n'
            f"val nll: {np.round(val_nll, 12)}" '\n'
            f"val mse: {np.round(val_mse, 12)}" '\n'
            f"best_val_nll: {np.round(best_val_nll, 12)}" '\n'
            f"epoch: {epoch + global_step}, best_epoch: {best_epoch}" '\n'
            f"current patience: {patience - ((epoch + global_step) - best_epoch)}" '\n'
        )

        # Early stopping.
        if patience - (epoch + global_step - best_epoch) < 0:
            break

        torch.save({'best_val_nll': best_val_nll,
                    'best_epoch': best_epoch,
                    'best_model_weights': best_model_weights,
                    'final_model_weights': model.state_dict(),
                    'final_optimizer_state_dict': optimizer.state_dict()
                    }, f'logs_and_weights/{NAME}/{NAME}.pth')

        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"val_NLL": (val_nll)}, checkpoint=checkpoint)


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train, iters_to_accumulate=1, beta=0.5):
    # Set model to training mode
    epoch_loss = 0.
    epoch_mean_squared_error = 0.
    mse = torch.nn.MSELoss()

    if train:
        model.train()

        if str(device) == 'cuda':
            print("using cuda")
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            with tqdm(dataloader, unit="batch") as tepoch:
                for i, (xb, yb) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(xb)
                        loss = beta_nll_loss(output[:, 0:2], output[:, 2:], yb[:, 0:2], beta)
                        gaussian_loss = loss_fn(output[:, 0:2], yb[:, 0:2], output[:, 2:])
                        loss = loss / iters_to_accumulate
                        mean_squared_error = mse(output[:, 0:2], yb[:, 0:2])

                    # Accumulates scaled gradients.
                    scaler.scale(loss).backward()

                    if (i + 1) % iters_to_accumulate == 0 or (i + 1) == len(dataloader):
                        # Gradient clipping.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += gaussian_loss.item() * xb.size(0)
                    epoch_mean_squared_error += mean_squared_error.item() * xb.size()[0]

                epoch_loss /= len(dataloader.sampler)
                epoch_mean_squared_error /= len(dataloader.sampler)
                tepoch.set_postfix(loss=epoch_loss)
                sleep(0.01)
        else:
            print("using cpu")
            with tqdm(dataloader, unit="batch") as tepoch:
                for i, (xb, yb) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    with torch.autocast(device_type='cpu', enabled=True):
                        output = model(xb)
                        loss = beta_nll_loss(output[:, 0:2], output[:, 2:], yb[:, 0:2], beta)

                    # Gradient clipping.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item() * xb.size(0)

                epoch_loss /= len(dataloader.sampler)
                tepoch.set_postfix(loss=epoch_loss)
                sleep(0.01)

    else:
        model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            # Iterate over data
            for xb, yb in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # forward
                with torch.set_grad_enabled(train):
                    output = model(xb, return_attention=False)
                    gaussian_loss = loss_fn(output[:, 0:2], yb[:, 0:2], output[:, 2:])
                    mean_squared_error = mse(output[:, 0:2], yb[:, 0:2])

                # statistics
                epoch_loss += gaussian_loss.item() * xb.size(0)
                epoch_mean_squared_error += mean_squared_error.item() * xb.size(0)

            epoch_loss /= len(dataloader.sampler)
            epoch_mean_squared_error /= len(dataloader.sampler)
            tepoch.set_postfix(loss=epoch_loss)
            sleep(0.01)

    return epoch_loss, epoch_mean_squared_error


def beta_nll_loss(mean, variance, target, beta):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)

    return torch.mean(loss)


if __name__ == '__main__':
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=1)