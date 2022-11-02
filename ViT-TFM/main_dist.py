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

import VisionTransformer as vit


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=500,
                        help='input batch size for validation (default: 500)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--use_amp', type=float, default=False,
                        help='use automatic mixed precision')
    # Misc
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
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

    # Preparing dataset
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
    # sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)

    dataloader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.use_cuda, 'shuffle': True}
    dataloaders = {'train': DataLoader(train_set, batch_size=args.batch_size, **dataloader_kwargs),
                   'val': DataLoader(val_set, batch_size=args.val_batch_size, **dataloader_kwargs)}

    # Create model
    vit_model = vit.VisionTransformer(dspl_size=104,
                                      patch_size=8,
                                      embed_dim=128,
                                      depth=12,
                                      n_heads=8,
                                      mlp_ratio=4.,
                                      p=0.1,
                                      attn_p=0.1,
                                      drop_path=0.1).double()
    n_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    # vit_model = nn.parallel.DistributedDataParallel(vit_model.cuda(), device_ids=[args.gpu])

    # Loss and optimizer
    loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # amp_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("am using cuda!!")
    else:
        device = torch.device('cpu')
        print("no cuda mehhh!!")

    NAME = "ViT-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
    fit(vit_model, loss, dataloaders, optimizer, device, writer, NAME, args.batch_size, args.patience)
    writer.close()


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience):
    best_val_rmse = np.inf
    best_epoch = -1
    best_model_weights = {}
    optimizer_state_dict_in_best_epoch = {}
    model.to(device)
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

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
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            optimizer_state_dict_in_best_epoch = copy.deepcopy(optimizer.state_dict())

        # Early stopping
        print(
            f"best_val_rmse: {np.round(best_val_rmse, 6)}, epoch: {epoch}, best_epoch: {best_epoch}, current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    torch.save({
        'best_epoch': best_epoch,
        'best_model_weights': best_model_weights,
        'optimizer_state_dict_in_best_epoch': optimizer_state_dict_in_best_epoch,
        'best_val_rmse': best_val_rmse
    },
        f'logs_and_weights/{NAME}/{NAME}_best_val_rmse_{np.round(best_val_rmse, 6)}.pth')


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


def init_distributed_model(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode.')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank
                                         )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)


if __name__ == '__main__':
    main()
