from time import sleep
from pytorch_pretrained_vit import ViT
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import argparse
import copy
import datetime
from datetime import datetime
import h5py
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import VisionTransformer_working as vit
sys.path.append('../scripts/')
from MultiTask import append_predictions_and_targets, dtma, dda


def execute():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--noise_level', type=float, default=1e-4,
                        help='Gaussian noise level to corrupt data with')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=2,
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--epochs', type=int, default=19,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--use_amp', type=bool, default=True,
                        help='use automatic mixed precision')
    # Misc
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers per GPU (default: 4')
    parser.add_argument('--continue_training', type=bool, default=True,
                        help='continue training given a checkpoint')
    parser.add_argument('--use_multi_task', type=bool, default=False,
                        help='optimize multi-task objective')
    parser.add_argument('--use_pretrained_model', type=bool, default=False,
                        help='use a ViT backbone')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Prepare training and validation dataset
    dspl_train = np.array(h5py.File('../data/Training data/resolution_104/allDisplacements.h5', 'r')['dspl'])
    trac_train = np.array(h5py.File('../data/Training data/resolution_104/allTractions.h5', 'r')['trac'])
    dspl_train = np.moveaxis(np.concatenate([dspl_train[i] for i in range(dspl_train.shape[0])], axis=0), 3, 1)
    trac_train = np.moveaxis(np.concatenate([trac_train[i] for i in range(trac_train.shape[0])], axis=0), 3, 1)
    X_train = torch.from_numpy(dspl_train).float()
    cov = [[args.noise_level ** 2, 0], [0, args.noise_level ** 2]]
    X_train_noise = np.transpose(np.random.multivariate_normal(np.array([0, 0]), cov, (X_train.shape[0], X_train.shape[2], X_train.shape[3])), (0, 3, 2, 1))
    X_train_noise = X_train + X_train_noise
    Y_train = torch.from_numpy(trac_train).float()
    train_set = TensorDataset(X_train_noise.float(), Y_train)

    dspl_val = np.array(h5py.File('../data/Validation data/resolution_104/allDisplacements.h5', 'r')['dspl'])
    trac_val = np.array(h5py.File('../data/Validation data/resolution_104/allTractions.h5', 'r')['trac'])
    dspl_val = np.moveaxis(np.concatenate([dspl_val[i] for i in range(dspl_val.shape[0])], axis=0), 3, 1)
    trac_val = np.moveaxis(np.concatenate([trac_val[i] for i in range(trac_val.shape[0])], axis=0), 3, 1)
    X_val = torch.from_numpy(dspl_val).float()
    X_val_noise = np.transpose(np.random.multivariate_normal(np.array([0, 0]), cov, (X_val.shape[0], X_val.shape[2], X_val.shape[3])), (0, 3, 2, 1))
    X_val_noise = X_val + X_val_noise
    Y_val = torch.from_numpy(trac_val).float()
    val_set = TensorDataset(X_val_noise.float(), Y_val)

    dataloader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.use_cuda, 'shuffle': True}
    dataloaders = {'train': DataLoader(train_set, batch_size=args.batch_size, **dataloader_kwargs),
                   'val': DataLoader(val_set, batch_size=args.val_batch_size, **dataloader_kwargs)}

    # Create model
    if args.use_pretrained_model:
        model_pretrained = ViT(name='B_16_imagenet1k', pretrained=True, in_channels=2, num_classes=10).float()
        vit_model = vit.PretrainedVit(model_pretrained=model_pretrained).float()
    else:
        vit_model = vit.VisionTransformer(dspl_size=104,
                                          patch_size=8,
                                          embed_dim=128,
                                          depth=12,
                                          n_heads=8,
                                          mlp_ratio=4.,
                                          p=0.05,
                                          attn_p=0.,
                                          drop_path=0.).float()

    n_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"Number of model parameters to optimize: {n_params}")

    # Specify device
    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Running on device: {device}")

    # Loss and optimizer
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)

    if args.continue_training:
        NAME = 'ViT_segmentation-2023-Jan-15 14:27:55'
        writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
        checkpoint = torch.load(f'logs_and_weights/{NAME}/{NAME}.pth')
        vit_model.load_state_dict(checkpoint['final_model_weights'], strict=True)
        vit_model.to(device)
        optimizer.load_state_dict(checkpoint['final_optimizer_state_dict'])
        event_acc = EventAccumulator(f'logs_and_weights/{NAME}')
        event_acc.Reload()
        global_step = event_acc.Scalars(tag='train_loss')[-1].step

    else:
        NAME = "ViT_segmentation-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
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
        args.epochs,
        args.patience,
        global_step,
        checkpoint=checkpoint,
        scheduler=None)

    writer.flush()
    writer.close()


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience, global_step, checkpoint=None, scheduler=None):
    if checkpoint is not None:
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        best_model_weights = checkpoint['best_model_weights']
    else:
        best_val_loss = np.inf
        best_epoch = 0
        best_model_weights = None

    model.to(device)
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, loss_fn, dataloaders['train'], device, epoch + global_step, optimizer, train=True)
        val_loss = run_epoch(model, loss_fn, dataloaders['val'], device, epoch + global_step, optimizer=None, train=False)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar('train_loss', train_loss, epoch + global_step)
        writer.add_scalar('val_loss', val_loss, epoch + global_step)

        # Save best weights.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + global_step
            best_model_weights = model.state_dict()

        print(
            '\n'
            f"Epoch: {epoch + global_step}/{max_epochs + global_step}" '\n'
            f"train loss: {np.round(train_loss, 12)}" '\n'
            f"val loss: {np.round(val_loss, 12)}" '\n'
            f"best_val_loss: {np.round(best_val_loss, 12)}" '\n'
            f"epoch: {epoch + global_step}, best_epoch: {best_epoch}" '\n'
            f"current patience: {patience - ((epoch + global_step) - best_epoch)}" '\n'
        )

        # Early stopping.
        if (epoch + global_step) - (best_epoch + global_step) >= patience:
            break

        torch.save({'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'best_model_weights': best_model_weights,
                    'final_model_weights': model.state_dict(),
                    'final_optimizer_state_dict': optimizer.state_dict()
                    }, f'logs_and_weights/{NAME}/{NAME}.pth')


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train, iters_to_accumulate=8):
    # Set model to training mode
    epoch_loss = 0.

    if train:
        model.train()

        if str(device) == 'cuda':
            print("using cuda")
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            with tqdm(dataloader, unit="batch") as tepoch:
                for i, (xb, yb) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    xb = xb.to(device, non_blocking=True)
                    yb = torch.where(yb[:, 2:] > 0, 1, 0)
                    yb = yb.to(device, non_blocking=True)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(xb)
                        loss = loss_fn(output, yb.float())
                        loss = loss / iters_to_accumulate

                    # Accumulates scaled gradients.
                    scaler.scale(loss).backward()

                    if (i + 1) % iters_to_accumulate == 0 or (i + 1) == len(dataloader):
                        # Gradient clipping.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item()

                epoch_loss /= len(dataloader.dataset)
                tepoch.set_postfix(loss=epoch_loss)
                sleep(0.01)
        else:
            print("using cpu")
            with tqdm(dataloader, unit="batch") as tepoch:
                for i, (xb, yb) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    xb = xb.to(device, non_blocking=True)
                    yb = torch.where(yb[:, 2:] > 0, 1, 0)
                    yb = yb.to(device, non_blocking=True)

                    with torch.autocast(device_type='cpu', enabled=True):
                        output = model(xb)
                        loss = loss_fn(output, yb)

                    # Gradient clipping.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item()

                epoch_loss /= len(dataloader.dataset)
                tepoch.set_postfix(loss=epoch_loss)
                sleep(0.01)

    else:
        model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            # Iterate over data
            for xb, yb in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                xb = xb.to(device, non_blocking=True)
                yb = torch.where(yb[:, 2:] > 0, 1, 0)
                yb = yb.to(device, non_blocking=True)

                # forward
                with torch.set_grad_enabled(train):
                    output = model(xb)
                    loss = loss_fn(output, yb.float())

                    # backward + optimize if in training phase
                    if train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                        optimizer.step()

                # statistics
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader.dataset)
            tepoch.set_postfix(loss=epoch_loss)
            sleep(0.01)

    return epoch_loss


if __name__ == '__main__':
    execute()