import VisionTransformer as vit

import argparse
import copy
import datetime
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
from time import sleep
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for validation (default: 500)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='WD',
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.val_batch_size}
    if use_cuda:
        num_workers = 4 * torch.cuda.device_count()
        cuda_kwargs = {'num_workers': num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    dspl_train = h5py.File('data/displacements_25.h5')["data"]
    trac_train = h5py.File('data/tractions_25.h5')["data"]
    dspl_val = h5py.File('data/displacements_5.h5')["data"]
    trac_val = h5py.File('data/tractions_5.h5')["data"]

    dspl_train = np.moveaxis(np.array(dspl_train), 3, 1)
    trac_train = np.moveaxis(np.array(trac_train), 3, 1)
    dspl_val = np.moveaxis(np.array(dspl_val), 3, 1)
    trac_val = np.moveaxis(np.array(trac_val), 3, 1)

    X_train = torch.from_numpy(dspl_train).double()
    Y_train = torch.from_numpy(trac_train).double()
    X_val = torch.from_numpy(dspl_val).double()
    Y_val = torch.from_numpy(trac_val).double()

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_set, **train_kwargs)
    dataloaders['val'] = DataLoader(val_set, **val_kwargs)

    vit_model = vit.VisionTransformer(dspl_size=104,
                                      patch_size=8,
                                      embed_dim=128,
                                      depth=12,
                                      n_heads=8,
                                      mlp_ratio=4.,
                                      p=0.1,
                                      attn_p=0.1,
                                      drop_path=0.1).double()

    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(vit_model.parameters(), args.lr, args.weight_decay)

    NAME = "ViT-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='{}'.format(NAME))
    fit(vit_model, loss, dataloaders, optimizer, device, writer, NAME, args.batch_size, args.patience)


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience):
    best_val_rmse = np.inf
    best_epoch = -1
    best_model_weights = {}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_rmse = run_epoch(model, loss_fn, dataloaders['train'], device, epoch, optimizer, train=True)
        val_loss, val_rmse = run_epoch(model, loss_fn, dataloaders['val'], device, epoch, optimizer=None, train=False)
        print(
            f"Epoch {epoch}/{max_epochs}, train_loss: {train_loss}, train_rmse: {train_rmse:.3f}")

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_rmse', train_rmse, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)

        # Save best weights
        if val_rmse < best_val_rmse:
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())

        # Early stopping
        print(
            f"best train_rmse: {best_val_rmse:.3f}, epoch: {epoch}, best_epoch: {best_epoch}, current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    torch.save(best_model_weights, f'{NAME}_best_train_rmse_{np.round(best_val_rmse, 6)}.pth')


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


if __name__ == '__main__':
    main()
