import os
import torch
import torch.nn as nn
import torch.distributed as dist
from utils.metrics import calculate_metrics


def init_dist_mode(local_rank):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    rank = int(os.environ['RANK'])
    # print(f"rank: {rank}")
    world_size = int(os.environ['WORLD_SIZE'])
    # print(f"world_size: {world_size}")
    local_rank = local_rank
    # print(f"local_rank: {local_rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    torch.cuda.set_device(local_rank)

    dist.barrier()


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def calculate_global_accuracy_and_loss(count, total, loss, world_size):
    # Converts the count and total of the local GPU to tensor
    local_metrics = torch.tensor([count, total, loss], dtype=torch.float64, device='cuda')
    # Wait for all GPUs to finish before calculating the global metrics
    dist.barrier()
    # Sum the count and total of all GPUs together
    dist.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
    # Calculate the global accuracy and loss
    local_metrics = local_metrics.tolist()
    all_count = int(local_metrics[0])
    all_total = int(local_metrics[1])
    all_loss = local_metrics[2] / world_size
    if all_count == 0:
        acc = 0
    else:
        acc = all_count / all_total
    return acc, all_loss


def calculate_global_metrics(preds_all, labels_all, world_size):
    # Calculate the local accuracy and loss
    oa, kappa, f1, pr, re = calculate_metrics(preds_all, labels_all)
    # Converts the count and total of the local GPU to tensor
    local_metrics = torch.tensor([oa, kappa, f1, pr, re], dtype=torch.float64, device='cuda')
    # Wait for all GPUs to finish before calculating the global metrics
    dist.barrier()
    # Sum the count and total of all GPUs together
    dist.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
    # Calculate the global accuracy and loss
    local_metrics = local_metrics.tolist()
    oa = local_metrics[0] / world_size
    kappa = local_metrics[1] / world_size
    f1 = local_metrics[2] / world_size
    pr = local_metrics[3] / world_size
    re = local_metrics[4] / world_size

    return oa, kappa, f1, pr, re

if __name__ == '__main__':
    ckpt_path = '/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/YPC/TGRS/checkpoints/MSCSC_China/' \
               'epoch50_lr0.04_batchsize256_patchsize11/best_acc.pth'
    # ckpt_path = '/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/YPC/TGRS_overall/checkpoints/MSCSC_China/' \
    #             'epoch100_lr0.04_batchsize32_patchsize11/best_acc.pth'
    checkpoint = torch.load(ckpt_path)
    print(checkpoint['classifier_state_dict'])
