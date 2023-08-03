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

import VisionTransformer_working_for_DPT as vit


def execute():
    # Training settings
    parser = argparse.ArgumentParser(description='TFM-ViT')
    parser.add_argument('--noise_percentage', type=float, default=0.5,
                        help='Percentage to multiply average variance over all training displacement fields with')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=50,
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
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers per GPU (default: 4')
    parser.add_argument('--continue_training', type=bool, default=False,
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
    # Train
    #dspl_train_2 = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/max5000/allDisplacements_final_max5000.h5', 'r')['dspl'], dtype="float32")
    #dspl_train_2_full = np.moveaxis(np.concatenate([dspl_train_2[i] for i in range(dspl_train_2.shape[0])], axis=0), 3, 1)
    #del dspl_train_2

    dspl = np.moveaxis(np.load('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm_new/data/train/displacements.npy'), 3, 1)
    sigma_bar = args.noise_percentage * np.mean(np.var(dspl, axis=(1, 2, 3)))
    print(f'percentage for noise level: {args.noise_percentage}')
    print(f'sigma_bar: {sigma_bar}')
    cov = [[sigma_bar, 0], [0, sigma_bar]]
    for i, x in tqdm(enumerate(dspl), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl[i] = x + noise
    X_train = torch.from_numpy(dspl).float()
    del dspl
    print(f'X_train_full.shape is {X_train.shape}')

    #trac_train_2 = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Training data/resolution_104/max5000/allTractions_final_max5000.h5', 'r')['trac'], dtype="float32")
   # trac_train_2_full = np.moveaxis(np.concatenate([trac_train_2[i] for i in range(trac_train_2.shape[0])], axis=0), 3, 1)
    #del trac_train_2

    trac = np.moveaxis(np.load('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm_new/data/train/tractions.npy'), 3, 1)
    Y_train = torch.from_numpy(trac).float()
    del trac
    train_dataset = TensorDataset(X_train.float(), Y_train)
    del X_train, Y_train
    print("Train set ready")

    # Val
    #dspl_val = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/max5000/allDisplacements_final_max5000.h5', 'r')['dspl'], dtype="float32")
    dspl_val = np.moveaxis(np.load('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm_new/data/validation/displacements.npy'), 3, 1)
    #trac_val = np.array(h5py.File('/home/alexrichard/PycharmProjects/UQ_DL-TFM/ViT-TFM/data/Validation data/resolution_104/max5000/allTractions_final_max5000.h5', 'r')['trac'], dtype="float32")
    trac_val = np.moveaxis(np.load('/home/alexrichard/PycharmProjects/UQ_DL-TFM/mltfm_new/data/validation/tractions.npy'), 3, 1)
    #dspl_val_full = np.moveaxis(np.concatenate([dspl_val[i] for i in range(dspl_val.shape[0])], axis=0), 3, 1)
    #del dspl_val
    #trac_val_full = np.moveaxis(np.concatenate([trac_val[i] for i in range(trac_val.shape[0])], axis=0), 3, 1)
    #del trac_val

    sigma_bar_val = args.noise_percentage * np.mean(np.var(dspl_val, axis=(1, 2, 3)))
    print(f'percentage for noise level: {args.noise_percentage}')
    print(f'sigma_bar_val: {sigma_bar_val}')
    cov = [[sigma_bar_val, 0], [0, sigma_bar_val]]
    for i, x in tqdm(enumerate(dspl_val), desc='noised'):
        noise = np.transpose(np.random.default_rng().multivariate_normal(mean=[0, 0], cov=cov, size=(104, 104)))
        dspl_val[i] = x + noise
    X_val = torch.from_numpy(dspl_val).float()
    Y_val = torch.from_numpy(trac_val).float()
    val_dataset = TensorDataset(X_val.float(), Y_val)
    del X_val, Y_val

    dataloader_kwargs = {'num_workers': args.num_workers, 'pin_memory': args.use_cuda, 'shuffle': True}
    dataloaders = {'train': DataLoader(train_dataset, batch_size=args.batch_size, **dataloader_kwargs),
                   'val': DataLoader(val_dataset, batch_size=args.batch_size, **dataloader_kwargs)}

    # Create model
    if args.use_pretrained_model:
        #model_pretrained = ViT(name='B_16_imagenet1k', pretrained=True, in_channels=2, num_classes=10).float()
        #vit_model = vit.PretrainedVit(model_pretrained=model_pretrained).float()
        checkpoint = torch.load(f'logs_and_weights/ViT_with_shifted_patch-2023-Jan-15 22:08:39/ViT_with_shifted_patch-2023-Jan-15 22:08:39.pth')
        vit_intermediate = vit.VisionTransformer(dspl_size=104,
                              patch_size=8,
                              embed_dim=128,
                              depth=12,
                              n_heads=8,
                              mlp_ratio=4.,
                              p=0.05,
                              attn_p=0.,
                              drop_path=0.).float()
        vit_intermediate.load_state_dict(checkpoint['best_model_weights'], strict=True)
        vit_model = vit.VisionTransformer3(dspl_size=104,
                                          patch_size=8,
                                          embed_dim=128,
                                          depth=6,
                                          n_heads=4,
                                          mlp_ratio=4.,
                                          p=0.05,
                                          attn_p=0.,
                                          drop_path=0.).float()
        print("Partly trained model created")
    else:
        vit_model = vit.VisionTransformer(dspl_size=104,
                                          patch_size=8,
                                          embed_dim=128,
                                          depth=6,
                                          n_heads=4,
                                          mlp_ratio=1.0,
                                          qkv_bias=False,
                                          p=0.1,
                                          attn_p=0.1,
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
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(vit_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
    layout = {
        "Plot": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]]
        },
    }

    if args.continue_training:
        NAME = 'ViT-final_2023-May-16 19:56:24'
        writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
        writer.add_custom_scalars(layout)
        checkpoint = torch.load(f'logs_and_weights/{NAME}/{NAME}.pth')
        vit_model.load_state_dict(checkpoint['final_model_weights'], strict=True)
        vit_model.to(device)
        optimizer.load_state_dict(checkpoint['final_optimizer_state_dict'])
        event_acc = EventAccumulator(f'logs_and_weights/{NAME}')
        event_acc.Reload()
        global_step = event_acc.Scalars(tag='train_loss')[-1].step

    else:
        NAME = "ViT-final_noise:0.5{:%Y-%b-%d %H:%M:%S}".format(datetime.now())
        writer = SummaryWriter(log_dir=f'logs_and_weights/{NAME}')
        writer.add_custom_scalars(layout)
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
        scheduler=scheduler)

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
        train_loss = run_epoch(model, loss_fn, dataloaders['train'], device, epoch + global_step, optimizer, train=True, iters_to_accumulate=1)
        val_loss = run_epoch(model, loss_fn, dataloaders['val'], device, epoch + global_step, optimizer=None, train=False, iters_to_accumulate=1)
        if scheduler is not None:
            print(f"Learning rate: {scheduler.get_last_lr()}")
            scheduler.step()

        writer.add_scalar('train_loss', train_loss, epoch + global_step)
        writer.add_scalar('val_loss', val_loss, epoch + global_step)
        writer.add_scalar('loss/train', train_loss, epoch + global_step)
        writer.add_scalar('loss/validation', val_loss, epoch + global_step)

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
        if patience - (epoch + global_step - best_epoch) < 0:
            break

        torch.save({'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'best_model_weights': best_model_weights,
                    'final_model_weights': model.state_dict(),
                    'final_optimizer_state_dict': optimizer.state_dict()
                    }, f'logs_and_weights/{NAME}/{NAME}.pth')


def run_epoch(model, loss_fn, dataloader, device, epoch, optimizer, train, iters_to_accumulate=1):
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
                    #if i == 1:
                    #    print(f'xb.shape is {xb.shape}')
                    #    print(f'yb.shape is {yb.shape}')
                    #xb_90 = torchvision.transforms.functional.rotate(xb, angle=90)
                    #xb_180 = torchvision.transforms.functional.rotate(xb, angle=180)
                    #xb_270 = torchvision.transforms.functional.rotate(xb, angle=270)
                    #xb = torch.vstack((xb, xb_90, xb_180, xb_270))
                    xb = xb.to(device, non_blocking=True)

                    #yb_90 = torchvision.transforms.functional.rotate(yb, angle=90)
                    #yb_180 = torchvision.transforms.functional.rotate(yb, angle=180)
                    #yb_270 = torchvision.transforms.functional.rotate(yb, angle=270)
                    #yb = torch.vstack((yb, yb_90, yb_180, yb_270))
                    yb = yb.to(device, non_blocking=True)

                    #if i == 1:
                    #    print(f'xb.shape after augm is {xb.shape}')
                    #    print(f'yb.shape after augm is {yb.shape}')

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        output = model(xb)
                        loss = loss_fn(output, yb[:, 0:2])
                        loss = loss / iters_to_accumulate

                    # Accumulates scaled gradients.
                    scaler.scale(loss).backward()

                    if (i + 1) % iters_to_accumulate == 0 or (i + 1) == len(dataloader):
                        # Gradient clipping.
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    # statistics
                    epoch_loss += loss.item() * xb.size(0)

                epoch_loss /= len(dataloader.sampler)
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
                        loss = loss_fn(output, yb[:, 0:2])

                    # Gradient clipping.
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
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
                    loss = loss_fn(output, yb[:, 0:2])

                # statistics
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataloader.sampler)
            tepoch.set_postfix(loss=epoch_loss)
            sleep(0.01)

    return epoch_loss


if __name__ == '__main__':
    execute()