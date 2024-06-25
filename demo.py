import os
import argparse
import torch.optim as optim
import torch.autograd
from configs import Configs as cfgs
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.seed import set_seed
from model.MSCSC import MSCSCNet, Classifier, ProjHead
from utils.loss import FocalLoss, ModifiedDINOLoss
import utils.ddp as ddp
from torch.distributed.elastic.multiprocessing.errors import record


# @record
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.device

    # world_size = int(os.environ['WORLD_SIZE'])
    # ddp.init_dist_mode(local_rank)  # initialize distributed training
    set_seed(cfgs.random_seed)
    # if local_rank == 0:
    print(f"seed: {cfgs.random_seed}, hi_feats: {cfgs.hi_feats}, iter_num: {cfgs.iter_num},"
          f" labeled_batch_size: {cfgs.train_labeled_batch_size},"
          f" unlabeled_batch_size: {cfgs.train_unlabeled_batch_size}, lr: {cfgs.lr}, patch_size: {cfgs.patch_size},"
          f" momentum: {cfgs.momentum}, weight_decay: {cfgs.weight_decay}.")



    student = MSCSCNet(cfgs.in_feats, cfgs.hi_feats, cfgs.iter_num).cuda()
    teacher = MSCSCNet(cfgs.in_feats, cfgs.hi_feats, cfgs.iter_num).cuda()
    classifier = Classifier(cfgs.hi_feats, cfgs.patch_size).cuda()
    projhead_student = ProjHead(in_dim=cfgs.hi_feats * (cfgs.patch_size - 4) ** 2, out_dim=cfgs.out_dim).cuda()
    projhead_teacher = ProjHead(in_dim=cfgs.hi_feats * (cfgs.patch_size - 4) ** 2, out_dim=cfgs.out_dim).cuda()

    criterion_sl = FocalLoss().cuda()
    criterion_ssl = ModifiedDINOLoss(cfgs.out_dim, cfgs.warmup_teacher_temp, cfgs.teacher_temp,
                                     cfgs.warmup_teacher_temp_epochs, cfgs.train_ssl_epochs,
                                     gamma=cfgs.gamma).cuda()
    optimizer = optim.SGD(
        list(student.parameters()) + list(projhead_student.parameters()) + list(classifier.parameters()), lr=cfgs.lr,
        momentum=cfgs.momentum, weight_decay=cfgs.weight_decay)

    trainer = Trainer(student, teacher, classifier, projhead_student, projhead_teacher, criterion_sl, criterion_ssl,
                      optimizer, None, cfgs)
    torch.autograd.set_detect_anomaly(True)
    trainer.train()

    # dist.barrier()

    evaluator = Evaluator(student, classifier, cfgs)
    evaluator.evaluate()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    #
    # local_rank = args.local_rank

    main()

    # main(local_rank)
