import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import utils.ddp as ddp
from dataloader.data_loader import HSICD_Dataset
from model.MSCSC import MSCSCNet, Classifier
from torch.utils.data import DataLoader
from utils.visualization import predict2img
from utils.metrics import calculate_metrics
from utils.ddp import calculate_global_metrics
from utils.logger import Logger
from utils.metrics import calculate_metrics
from configs import Configs as cfgs


class Evaluator:
    def __init__(self, student, classifier, configs):
        # self.local_rank = local_rank
        # self.world_size = world_size

        self.student = student
        self.classifier = classifier

        self.dataset_name = configs.dataset_name
        self.patch_size = configs.patch_size

        self.epochs = configs.epochs
        self.lr = configs.lr
        self.train_labeled_batch_size = configs.train_labeled_batch_size
        self.test_batch_size = configs.test_batch_size
        # self.test_batch_size = configs.test_batch_size // self.world_size
        self.test_num_workers = configs.test_num_workers
        self.resume_path = configs.resume_path
        self.save_results_folder = configs.save_results_folder
        if not os.path.exists(os.path.join(self.save_results_folder)):
            os.makedirs(os.path.join(self.save_results_folder))

        self.log_path = configs.log_path + f"/epoch{self.epochs}_lr{self.lr}_batchsize{self.train_labeled_batch_size}" \
                                           f"_patchsize{self.patch_size}.txt"
        self.logger = Logger(os.path.join(self.log_path))

    def resume_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

    def evaluate(self):
        test_data = HSICD_Dataset(self.dataset_name, mode='test', patch_size=self.patch_size)
        img_gt = test_data.img_gt
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.test_batch_size, shuffle=False,
                                 num_workers=self.test_num_workers, pin_memory=True)

        self.resume_ckpt(self.resume_path)

        # if ddp.has_batchnorms(self.student):
        #     self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
        #     self.classifier = nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier)
        #
        # self.student = nn.parallel.DistributedDataParallel(self.student, device_ids=[self.local_rank],
        #                                                    output_device=self.local_rank)
        # self.classifier = nn.parallel.DistributedDataParallel(self.classifier, device_ids=[self.local_rank],
        #                                                       output_device=self.local_rank)

        self.student.eval()
        self.classifier.eval()

        preds_all = []
        labels_all = []
        pos_all = []

        # if self.local_rank == 0:
        print("****************Start Testing****************\n")

        batch_total_time = 0
        start_total_time = time.time()
        with torch.no_grad():
            for batch_idx, (img_t1, img_t2, label, pos) in enumerate(test_loader):
                batch_start_time = time.time()

                img_t1 = img_t1.cuda()
                img_t2 = img_t2.cuda()
                label = label.cuda()

                logits = self.classifier(self.student(img_t1), self.student(img_t2))

                # torch.cuda.synchronize()

                batch_total_time += time.time() - batch_start_time
                preds = torch.argmax(logits, dim=1)

                preds_all.append(preds.detach().cpu().numpy())
                labels_all.append(label.detach().cpu().numpy())
                pos_all.append(pos.detach().cpu().numpy())

        infer_batch_time = batch_total_time
        infer_total_time = time.time() - start_total_time

        # if self.local_rank == 0:
        print('****************Testing Completed!****************\n')

        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        pos_all = np.concatenate(pos_all, axis=0)

        oa, kappa, f1, pr, re, aa = calculate_metrics(preds_all, labels_all)

        # if self.local_rank == 0:
        print(f"Infer total time: {infer_total_time:.4f}s || Infer model time: {infer_batch_time:.4f}s ||"
              f" OA: {(oa * 100):.2f}% || Kappa: {kappa:.4f} || F1: {(f1 * 100):.2f}% || Pr: {(pr * 100):.2f}% ||"
              f" Re: {(re * 100):.2f}% || AA: {(aa * 100):.2f}%\n")

        self.logger.get_test_logs(infer_total_time, infer_batch_time, oa, kappa, f1, pr, re, aa)

        predict2img(preds_all, img_gt, pos_all, self.dataset_name, self.save_results_folder)


if __name__ == '__main__':
    student = MSCSCNet(cfgs.in_feats, cfgs.hi_feats, cfgs.iter_num).cuda()
    classifier = Classifier(cfgs.hi_feats, cfgs.patch_size).cuda()
    evaluator = Evaluator(student, classifier, cfgs)
