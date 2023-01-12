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

import VisionTransformer_working as vit
from MultiTask import multi_task_loss, dtma, dda, append_predictions_and_targets


def execute():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping.')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='initial learning rate (default: 0.0005)')
    parser.add_argument('--use_amp', type=bool, default=True,
                        help='use automatic mixed precision')
    # Misc
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of data loading workers per GPU (default: 4')
    parser.add_argument('--continue_training', type=bool, default=False,
                        help='continue training given a checkpoint')
    parser.add_argument('--use_multi_task', type=bool, default=True,
                        help='optimize multi-task objective')
    args = parser.parse_args()

    # If possible, setup distributed training and seeds
    # init_distributed_model(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Preparing dataset
    dspl = np.array(h5py.File(f'../data/extended data/allDisplacements.h5')['dspl'])
    trac_separated = np.array(h5py.File(f'../data/extended data/allTractions.h5')['trac'])

    dspl = np.moveaxis(np.concatenate([dspl[i] for i in range(dspl.shape[0])], axis=0), 3, 1)
    trac_separated = np.moveaxis(np.concatenate([trac_separated[i] for i in range(trac_separated.shape[0])], axis=0), 3, 1)

    dspl_train = dspl[0:20, :, :, :]
    dspl_val = dspl[20:25, :, :, :]
    trac_train = trac_separated[0:20, :, :, :]
    trac_val = trac_separated[20:25, :, :, :]

    X_train = torch.from_numpy(dspl_train).float()
    Y_train = torch.from_numpy(trac_train).float()
    X_val = torch.from_numpy(dspl_val).float()
    Y_val = torch.from_numpy(trac_val).float()

    train_set = TensorDataset(X_train, Y_train)
    val_set = TensorDataset(X_val, Y_val)

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
                                      p=0.,
                                      attn_p=0.,
                                      drop_path=0.).float()

    n_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
    print(f"Number of model parameters to optimize: {n_params}")

    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Running on device: {device}")

    # Loss and optimizer
    if args.use_multi_task:
        loss = multi_task_loss
    else:
        loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, last_epoch=-1)

    NAME = "ViT-{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
    if args.continue_training:
        checkpoint = torch.load(
            'logs_and_weights/ViT-2023-Jan-08 11:21:19/ViT-2023-Jan-08 11:21:19_best_val_loss_0.003494746022.pth')
        vit.model = vit_model.load_state_dict(checkpoint['best_model_weights'])
        vit_model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict_in_best_epoch'])
    fit(vit_model, loss, dataloaders, optimizer, device, writer, NAME, args.epochs, args.patience, scheduler=None)
    writer.close()


def fit(model, loss_fn, dataloaders, optimizer, device, writer, NAME, max_epochs, patience, scheduler=None):
    best_val_loss = np.inf
    best_epoch = -1
    best_model_weights = {}
    optimizer_state_dict_in_best_epoch = {}
    model.to(device)
    example_input, example_value = next(iter(dataloaders['train']))
    writer.add_graph(model, example_input.to(device))

    for epoch in range(1, max_epochs + 1):
        train_loss, train_dtma, train_dda = run_epoch(model, loss_fn, dataloaders['train'], device, epoch, optimizer, train=True)
        val_loss, val_dtma, val_dda = run_epoch(model, loss_fn, dataloaders['val'], device, epoch, optimizer=None, train=False)
        if scheduler is not None:
            scheduler.step()

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_dtma', train_dtma, epoch)
        writer.add_scalar('train_dda', train_dda, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_dtma', val_dtma, epoch)
        writer.add_scalar('val_dda', val_dda, epoch)

        # Save best weights.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            optimizer_state_dict_in_best_epoch = copy.deepcopy(optimizer.state_dict())

        print(
            '\n'
            f"Epoch: {epoch}/{max_epochs}" '\n'
            f"train loss: {np.round(train_loss, 12)}" '\n'
            f"train dtma: {np.round(train_dtma, 12)}" '\n'
            f"train dda: {np.round(train_dda, 12)}" '\n'
            f"val loss: {np.round(val_loss, 12)}" '\n'
            f"val dtma: {np.round(val_dtma, 12)}" '\n'
            f"val dda: {np.round(val_dda, 12)}" '\n'
            f"best_val_loss: {np.round(best_val_loss, 12)}" '\n'
            f"epoch: {epoch}, best_epoch: {best_epoch}" '\n'
            f"current patience: {patience - (epoch - best_epoch)}" '\n'
        )

        # Early stopping.
        if epoch - best_epoch >= patience:
            break

    torch.save({
        'best_epoch': best_epoch,
        'best_model_weights': best_model_weights,
        'optimizer_state_dict_in_best_epoch': optimizer_state_dict_in_best_epoch,
    },
        f'logs_and_weights/{NAME}/{NAME}_best_val_loss_{np.round(best_val_loss, 12)}.pth')


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train):
    # Set model to training mode
    epoch_loss = 0.0
    epoch_dtma = 0.0
    epoch_dda = 0.0

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
                        loss = loss_fn(output, yb, device=device)
                        loss = loss / 1

                    # Accumulates scaled gradients.
                    scaler.scale(loss).backward()

                    if (i + 1) % 1 == 0 or (i + 1) == len(dataloader):
                        # Gradient clipping.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item()
                    appended_predictions, appended_targets = append_predictions_and_targets(output, yb, device=device)
                    epoch_dtma += dtma(appended_predictions, appended_targets, device=device).item()
                    epoch_dda += dda(appended_predictions, appended_targets, device=device).item()

                epoch_loss /= len(dataloader.dataset)
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
                        loss = loss_fn(output, yb, device=device)

                    # Gradient clipping.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item()
                    appended_predictions, appended_targets = append_predictions_and_targets(output, yb, device=device)
                    epoch_dtma += dtma(appended_predictions, appended_targets, device=device).item()
                    epoch_dda += dda(appended_predictions, appended_targets, device=device).item()

                epoch_loss /= len(dataloader.dataset)
                tepoch.set_postfix(loss=epoch_loss)
                sleep(0.01)

    else:
        model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            # Iterate over data
            for xb, yb in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                xb, yb = xb.to(device), yb.to(device)

                # forward
                with torch.set_grad_enabled(train):
                    output = model(xb)
                    loss = loss_fn(output, yb, device=device)

                    # backward + optimize if in training phase
                    if train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
                        optimizer.step()

                # statistics
                epoch_loss += loss.item()
                appended_predictions, appended_targets = append_predictions_and_targets(output, yb, device=device)
                epoch_dtma += dtma(appended_predictions, appended_targets, device=device).item()
                epoch_dda += dda(appended_predictions, appended_targets, device=device).item()

            epoch_loss /= len(dataloader.dataset)
            tepoch.set_postfix(loss=epoch_loss)
            sleep(0.01)

    return epoch_loss, epoch_dtma, epoch_dda


if __name__ == '__main__':
    execute()