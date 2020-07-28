import os
import socket
import time
import sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset, DistFieldSampler
from .data.figures import plt_slices
from . import models
from .models import narrow_cast, resample, Lag2Eul
from .utils import import_attr, load_model_state_dict


ckpt_link = 'checkpoint.pt'


def node_worker(args):
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node

    node = int(os.environ['SLURM_NODEID'])

    if args.gpus_per_node < 1:
        raise RuntimeError('GPU not found on node {}'.format(node))

    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):
    #device = torch.device('cuda', local_rank)
    #torch.cuda.device(device)  # env var recommended over this

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    device = torch.device('cuda', 0)

    rank = args.gpus_per_node * node + local_rank

    # Need randomness across processes, for sampler, augmentation, noise etc.
    # Note DDP broadcasts initial model states from rank 0
    torch.manual_seed(args.seed + rank)
    #torch.backends.cudnn.deterministic = True  # NOTE: test perf

    dist_init(rank, args)

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        pad=args.pad,
        scale_factor=args.scale_factor,
    )
    train_sampler = DistFieldSampler(train_dataset, shuffle=True,
                                     div_data=args.div_data,
                                     div_shuffle_dist=args.div_shuffle_dist)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batches,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=False,
            aug_shift=None,
            aug_add=None,
            aug_mul=None,
            crop=args.crop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop,
            crop_step=args.crop_step,
            pad=args.pad,
            scale_factor=args.scale_factor,
        )
        val_sampler = DistFieldSampler(val_dataset, shuffle=False,
                                       div_data=args.div_data,
                                       div_shuffle_dist=args.div_shuffle_dist)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batches,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True,
        )

    args.in_chan, args.out_chan = train_dataset.in_chan, train_dataset.tgt_chan

    model = import_attr(args.model, models.__name__, args.callback_at)
    model = model(sum(args.in_chan), sum(args.out_chan))
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device],
            process_group=dist.new_group())

    lag2eul = Lag2Eul()

    criterion = import_attr(args.criterion, nn.__name__, args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim.__name__, args.callback_at)
    lag_optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )
    eul_optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )
    lag_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        lag_optimizer, **args.scheduler_args)
    eul_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        eul_optimizer, **args.scheduler_args)

    if (args.load_state == ckpt_link and not os.path.isfile(ckpt_link)
            or not args.load_state):
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                m.weight.data.normal_(0.0, args.init_weight_std)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.affine:
                    # NOTE: dispersion from DCGAN, why?
                    m.weight.data.normal_(1.0, args.init_weight_std)
                    m.bias.data.fill_(0)

        if args.init_weight_std is not None:
            model.apply(init_weights)

        start_epoch = 0

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state['epoch']

        load_model_state_dict(model.module, state['model'],
                strict=args.load_state_strict)

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']

            print('state at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state), flush=True)

        del state

    torch.backends.cudnn.benchmark = True  # NOTE: test perf

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        pprint(vars(args))
        sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader, model, lag2eul, criterion,
            lag_optimizer, eul_optimizer, lag_scheduler, eul_scheduler,
            logger, device, args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader, model, lag2eul, criterion,
                logger, device, args)
            #epoch_loss = val_loss

        if args.reduce_lr_on_plateau:
            lag_scheduler.step(epoch_loss[0])
            eul_scheduler.step(epoch_loss[1])

        if rank == 0:
            logger.flush()

            if min_loss is None or torch.prod(epoch_loss) < torch.prod(min_loss):
                min_loss = epoch_loss

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }

            state_file = 'state_{}.pt'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()


def train(epoch, loader, model, lag2eul, criterion,
        lag_optimizer, eul_optimizer, lag_scheduler, eul_scheduler,
        logger, device, args):
    model.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)

    for i, (input, target) in enumerate(loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        if epoch == 0 and i == 0 and rank == 0:
            print('input.shape =', input.shape)
            print('output.shape =', output.shape)
            print('target.shape =', target.shape, flush=True)

        if (hasattr(model.module, 'scale_factor')
                and model.module.scale_factor != 1):
            input = resample(input, model.module.scale_factor, narrow=False)
        input, output, target = narrow_cast(input, output, target)

        lag_out, lag_tgt = output, target

        if i % 2 == 0:
            lag_loss = criterion(lag_out, lag_tgt)
            epoch_loss[0] += lag_loss.item()

            with torch.no_grad():
                eul_out, eul_tgt = lag2eul(lag_out, lag_tgt)

                eul_loss = criterion(eul_out, eul_tgt)
                epoch_loss[1] += eul_loss.item()

            lag_optimizer.zero_grad()
            lag_loss.backward()
            lag_optimizer.step()
            lag_grads = get_grads(model)
        else:
            with torch.no_grad():
                lag_loss = criterion(lag_out, lag_tgt)
                epoch_loss[0] += lag_loss.item()

            eul_out, eul_tgt = lag2eul(lag_out, lag_tgt)

            eul_loss = criterion(eul_out, eul_tgt)
            epoch_loss[1] += eul_loss.item()

            eul_optimizer.zero_grad()
            eul_loss.backward()
            eul_optimizer.step()
            eul_grads = get_grads(model)

        batch = epoch * len(loader) + i + 1
        if batch % args.log_interval == 0 and batch >= 2:
            dist.all_reduce(lag_loss)
            dist.all_reduce(eul_loss)
            lag_loss /= world_size
            eul_loss /= world_size
            if rank == 0:
                logger.add_scalar('loss/batch/train/lag', lag_loss.item(),
                                  global_step=batch)
                logger.add_scalar('loss/batch/train/eul', eul_loss.item(),
                                  global_step=batch)
                logger.add_scalar('loss/batch/train/lxe',
                                  lag_loss.item() * eul_loss.item(),
                                  global_step=batch)

                logger.add_scalar('grad/lag/first', lag_grads[0],
                                  global_step=batch)
                logger.add_scalar('grad/lag/last', lag_grads[-1],
                                  global_step=batch)
                logger.add_scalar('grad/eul/first', eul_grads[0],
                                  global_step=batch)
                logger.add_scalar('grad/eul/last', eul_grads[-1],
                                  global_step=batch)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/train/lag', epoch_loss[0],
                          global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/eul', epoch_loss[1],
                          global_step=epoch+1)
        logger.add_scalar('loss/epoch/train/lxe', epoch_loss.prod(),
                          global_step=epoch+1)

        fig = plt_slices(
            input[-1], lag_out[-1], lag_tgt[-1], lag_out[-1] - lag_tgt[-1],
                       eul_out[-1], eul_tgt[-1], eul_out[-1] - eul_tgt[-1],
            title=['in', 'lag_out', 'lag_tgt', 'lag_out - lag_tgt',
                         'eul_out', 'eul_tgt', 'eul_out - eul_tgt'],
        )
        logger.add_figure('fig/epoch/train', fig, global_step=epoch+1)
        fig.clf()

    return epoch_loss


def validate(epoch, loader, model, lag2eul, criterion, logger, device, args):
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)

    with torch.no_grad():
        for input, target in loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)

            if (hasattr(model.module, 'scale_factor')
                    and model.module.scale_factor != 1):
                input = resample(input, model.module.scale_factor, narrow=False)
            input, output, target = narrow_cast(input, output, target)

            lag_out, lag_tgt = output, target

            lag_loss = criterion(lag_out, lag_tgt)
            epoch_loss[0] += lag_loss.item()

            eul_out, eul_tgt = lag2eul(lag_out, lag_tgt)

            eul_loss = criterion(eul_out, eul_tgt)
            epoch_loss[1] += eul_loss.item()

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/val/lag', epoch_loss[0],
                          global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/eul', epoch_loss[1],
                          global_step=epoch+1)
        logger.add_scalar('loss/epoch/val/lxe', epoch_loss.prod(),
                          global_step=epoch+1)

        fig = plt_slices(
            input[-1], lag_out[-1], lag_tgt[-1], lag_out[-1] - lag_tgt[-1],
                       eul_out[-1], eul_tgt[-1], eul_out[-1] - eul_tgt[-1],
            title=['in', 'lag_out', 'lag_tgt', 'lag_out - lag_tgt',
                         'eul_out', 'eul_tgt', 'eul_out - eul_tgt'],
        )
        logger.add_figure('fig/epoch/val', fig, global_step=epoch+1)
        fig.clf()

    return epoch_loss


def dist_init(rank, args):
    dist_file = 'dist_addr'

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()

        args.dist_addr = 'tcp://{}:{}'.format(addr, port)

        with open(dist_file, mode='w') as f:
            f.write(args.dist_addr)

    if rank != 0:
        while not os.path.exists(dist_file):
            time.sleep(1)

        with open(dist_file, mode='r') as f:
            args.dist_addr = f.read()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    dist.barrier()

    if rank == 0:
        os.remove(dist_file)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """gradients of the weights of the first and the last layer
    """
    grads = list(p.grad for n, p in model.named_parameters()
                 if '.weight' in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm().item() for g in grads]
    return grads
