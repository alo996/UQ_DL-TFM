import argparse
import copy
import datetime
import os
from datetime import datetime
from time import sleep

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DPT import DPT


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--use_pretrained_model', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.05,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--use_amp', type=float, default=False,
                        help='use automatic mixed precision')
    # Misc
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed (default: 2)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of data loading workers per GPU (default: 4')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    # If possible, setup distributed training and seeds
    # init_distributed_model(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    device = torch.device('cpu')

    # Preparing dataset
    dspl = h5py.File('/Users/alex/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/displacements_25.h5')["data"]
    trac = h5py.File('/Users/alex/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/tractions_25.h5')["data"]

    dspl = np.moveaxis(np.array(dspl), 3, 1)
    trac = np.moveaxis(np.array(trac), 3, 1)

    dspl_train = dspl[0:24, :, :, :]
    dspl_val = dspl[24:, :, :, :]
    trac_train = trac[0:24, :, :, :]
    trac_val = trac[24:, :, :, :]

    X_train = torch.from_numpy(dspl_train).float()
    Y_train = torch.from_numpy(trac_train).float()
    X_val = torch.from_numpy(dspl_val).float()
    Y_val = torch.from_numpy(trac_val).float()

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)

    dataloader_kwargs = {'batch_size': args.batch_size,
                         'num_workers': args.num_workers,
                         'pin_memory': str(device) == 'cuda',
                         'shuffle': True}
    dataloaders = {'train': DataLoader(train_set, **dataloader_kwargs),
                   'val': DataLoader(val_set, **dataloader_kwargs)}
    # Create model
    dpt_model = DPT().float()
    n_params = sum(p.numel() for p in dpt_model.parameters() if p.requires_grad)
    print(f"number of parameters in dpt: {n_params}")

    # Loss and optimizer
    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(dpt_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # amp_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=3, t_total=args.epochs)

    NAME = "DPT-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
    fit(dpt_model, loss, dataloaders, optimizer, device, writer, NAME, args.epochs, args.patience)
    writer.close()


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience):
    best_val_loss = np.inf
    best_epoch = -1
    best_model_weights = {}
    optimizer_state_dict_in_best_epoch = {}
    model.to(device)
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, loss_fn, dataloaders['train'], device, epoch, optimizer, train=True)
        val_loss = run_epoch(model, loss_fn, dataloaders['val'], device, epoch, optimizer=None, train=False)
        print(
            f"Epoch {epoch}/{max_epochs}, train_loss: {train_loss}")

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            optimizer_state_dict_in_best_epoch = copy.deepcopy(optimizer.state_dict())

        # Early stopping
        print(
            f"best_val_loss: {np.round(best_val_loss, 6)}, "
            f"epoch: {epoch}, best_epoch: {best_epoch}, "
            f"current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    torch.save({
        'best_epoch': best_epoch,
        'best_model_weights': best_model_weights,
        'optimizer_state_dict_in_best_epoch': optimizer_state_dict_in_best_epoch,
        'best_val_loss': best_val_loss
    },
        f'logs_and_weights/{NAME}/{NAME}_best_val_loss_{np.round(best_val_loss, 6)}.pth')


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train, visualize_attn=False):
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
            if visualize_attn:
                pred, attn_scores = model(xb)  # attn_scores: [attn_1, ..., attn_depth]
            else:
                pred = model(xb)
            if not train and visualize_attn:
                with torch.set_grad_enabled(not train):
                    attn_mat = torch.stack(attn_scores) # attn_mat.shape==(depth, samples, n_heads, n_patches**0.5, n_patches**0.5)
                    attn_mat = attn_mat.squeeze(1) # shape remains unchanged unless samples==1, then: (depth, n_heads, n_patches**0.5, n_patches**0.5)
                    attn_mat = torch.mean(attn_mat, dim=1) # if samples==1: attn_mat.shape==(depth, n_patches**0.5,n_patches**0.5)

            with torch.set_grad_enabled(train):
                print(pred.shape)
                print(pred.shape)
                loss = loss_fn(pred, yb)

                # backward + optimize if in training phase
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()

            # statistics
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader.dataset)
        tepoch.set_postfix(loss=epoch_loss)
        sleep(0.01)
    return epoch_loss


def inference_with_dropout(model, inputs, return_attn_scores=True):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    if return_attn_scores:
        return model(inputs)
    else:
        return model(inputs, return_attn_scores=False)


if __name__ == '__main__':
    main()
