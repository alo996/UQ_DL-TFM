import argparse
import copy
import datetime
from datetime import datetime
from time import sleep

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gc import collect

from DPT import DPT, init_weights


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--use_pretrained_model', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--patience', type=int, default=70,
                        help='Early stopping.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    parser.add_argument('--iters_to_accumulate', default=4,
                        help='quickly check a single pass')
    parser.add_argument('--dry_run', default=False,
                        help='quickly check a single pass')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.5,
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.000005,
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--use_amp', type=float, default=True,
                        help='use automatic mixed precision')
    # Misc
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 2)')
    parser.add_argument('--num_workers', type=int, default=4,
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

    # Prepare training and validation dataset.
    dspl = h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm/displacements_res_104_num_50000.h5')["data"]
    trac = h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm/tractions_res_104_num_50000.h5')["data"]

    dspl = np.moveaxis(np.array(dspl), 3, 1)
    trac = np.moveaxis(np.array(trac), 3, 1)

    dspl_train = dspl[0:2500, :, :, :]
    dspl_val = dspl[2500:2550, :, :, :]
    trac_train = trac[0:2500, :, :, :]
    trac_val = trac[2500:2550, :, :, :]

    X_train = torch.from_numpy(dspl_train).float()
    Y_train = torch.from_numpy(trac_train).float()
    X_val = torch.from_numpy(dspl_val).float()
    Y_val = torch.from_numpy(trac_val).float()

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)

    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': str(device) == 'cuda'}
    dataloaders = {
        'train': DataLoader(train_set, shuffle=True, **dataloader_kwargs),
        'val': DataLoader(val_set, shuffle=False, **dataloader_kwargs)}

    # Create model
    dpt_model = DPT(device=device)
    dpt_model.apply(init_weights)
    n_params = sum(p.numel() for p in dpt_model.parameters() if p.requires_grad)
    print(f"number of parameters in dpt: {n_params}")

    # Loss and optimizer
    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(dpt_model.parameters(), lr=args.lr, eps=1e-03, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

    NAME = "DPT-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
    collect()
    torch.cuda.empty_cache()
    fit(dpt_model, loss, dataloaders, optimizer, device, writer, NAME, args.epochs, args.patience, args.iters_to_accumulate, scheduler)
    writer.close()


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience, iters_to_accumulate, scheduler=None):
    best_val_loss = np.inf
    best_epoch = -1
    best_model_weights = {}
    train_loss = np.inf
    optimizer_state_dict_in_best_epoch = {}
    model = model.to(device)

    # Create Tensorboard graph
    model.eval()
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

    for epoch in range(1, max_epochs + 1):
        train_loss = run_train_epoch(model, loss_fn, dataloaders['train'], device, epoch, optimizer, iters_to_accumulate, use_amp=True)
        val_loss = run_inference_epoch(model, loss_fn, dataloaders['val'], device, epoch, use_amp=True, visualize_attn=False)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

        # Save best weights.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            optimizer_state_dict_in_best_epoch = copy.deepcopy(optimizer.state_dict())

        # Early stopping.
        print(
            '\n'
            f"Epoch: {epoch}/{max_epochs}" '\n'
            f"train loss: {np.round(train_loss, 12)}" '\n'
            f"best_val_loss: {np.round(best_val_loss, 12)}" '\n'
            f"epoch: {epoch}, best_epoch: {best_epoch}" '\n'
            f"current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    last_model_weights = copy.deepcopy(model.state_dict())

    torch.save({
        'best_epoch': best_epoch,
        'best_model_weights': best_model_weights,
        'last_model_weights': last_model_weights,
        'optimizer_state_dict_in_best_epoch': optimizer_state_dict_in_best_epoch,
        'loss': train_loss,
        'best_val_loss': best_val_loss
    },
        f'logs_and_weights/{NAME}/{NAME}_best_val_loss_{np.round(best_val_loss, 10)}.pth')


def run_train_epoch(model, loss_fn, dataloader, device, epoch, optimizer, iters_to_accumulate=1, use_amp=True):
    model.train()
    # Reset the gradients to None.
    optimizer.zero_grad(set_to_none=True)
    epoch_loss = 0.0

    if str(device) == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (xb, yb) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    output = model(xb)
                    assert output.dtype is torch.float16
                    loss = loss_fn(output, yb)
                    loss = loss / iters_to_accumulate
                    assert loss.dtype is torch.float32

                # Accumulates scaled gradients.
                scaler.scale(loss).backward()

                if (i+1) % 2 == 0 or (i+1) == len(dataloader):
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

    elif str(device) == 'cpu':
        with tqdm(dataloader, unit="batch") as tepoch:
            for i, (xb, yb) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad(set_to_none=True)

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.autocast(device_type='cpu', enabled=use_amp):
                    output = model(xb)
                    loss = loss_fn(output, yb)

                # Gradient clipping.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                optimizer.step()

                # statistics
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader.dataset)
            tepoch.set_postfix(loss=epoch_loss)
            sleep(0.01)

    return epoch_loss


def run_inference_epoch(model, loss_fn, dataloader, device, epoch, use_amp=True, visualize_attn=False):
    # Set model to evaluation mode.
    model.eval()
    epoch_loss = 0.0

    with tqdm(dataloader, unit="batch") as tepoch:
        # Iterate over data
        for xb, yb in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if str(device) == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    if visualize_attn:
                        pred, attn_scores = model(xb)  # attn_scores: [attn_1, ..., attn_depth]
                        assert pred.dtype is torch.float16
                    else:
                        pred = model(xb)
                        assert pred.dtype is torch.float16
                    loss = loss_fn(pred, yb)
                    assert loss.dtype is torch.float32

            elif str(device) == 'cpu':
                with torch.autocast(device_type='cpu', dtype=torch.float16, enabled=use_amp):
                    if visualize_attn:
                        pred, attn_scores = model(xb)  # attn_scores: [attn_1, ..., attn_depth]
                        assert pred.dtype is torch.float16
                    else:
                        pred = model(xb)
                        assert pred.dtype is torch.float16
                    loss = loss_fn(pred, yb)
                    assert loss.dtype is torch.float32

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
