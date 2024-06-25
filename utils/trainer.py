import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import utils.ddp as ddp
import datetime
from torch import optim
from dataloader.data_loader import HSICD_Dataset
from torch.utils.data import DataLoader
from configs import Configs as cfgs
from utils.logger import Logger
import matplotlib.pyplot as plt
from utils.loss import FocalLoss, ModifiedDINOLoss
from utils.seed import set_seed
from model.MSCSC import MSCSCNet, Classifier, ProjHead


class Trainer:
    def __init__(self, student, teacher, classifier, projhead_student, projhead_teacher, criterion_sl, criterion_ssl,
                 optimizer, scheduler, configs):
        # self.local_rank = local_rank
        # self.world_size = world_size
        self.student = student
        self.teacher = teacher
        self.classifier = classifier
        self.projhead_student = projhead_student
        self.projhead_teacher = projhead_teacher
        self.criterion_sl = criterion_sl
        self.criterion_ssl = criterion_ssl
        # self.criterion = criterion
        # self.optimizer_sl = optim.SGD(list(self.student.parameters()) + list(self.classifier.parameters()),
        #                               lr=cfgs.lr, momentum=cfgs.momentum, weight_decay=cfgs.weight_decay)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataset_name = configs.dataset_name
        self.seed = configs.random_seed
        self.patch_size = configs.patch_size
        self.eta = configs.eta
        self.momentum_teacher = configs.momentum_teacher
        self.out_dim = configs.out_dim
        self.warmup_teacher_temp = configs.warmup_teacher_temp
        self.teacher_temp = configs.teacher_temp
        self.warmup_teacher_temp_epochs = configs.warmup_teacher_temp_epochs
        self.gamma = configs.gamma
        self.clip_grad = configs.clip_grad
        self.use_clip_grad = configs.use_clip_grad
        self.ssl_start = True

        self.train_labeled_batch_size = configs.train_labeled_batch_size
        # self.train_labeled_batch_size = configs.train_labeled_batch_size // world_size
        self.train_unlabeled_batch_size = configs.train_unlabeled_batch_size
        # self.train_unlabeled_batch_size = configs.train_unlabeled_batch_size // world_size
        self.val_batch_size = configs.val_batch_size
        # self.val_batch_size = configs.val_batch_size // world_size
        self.epochs = configs.epochs
        self.train_sl_epochs = cfgs.train_sl_epochs
        self.train_ssl_epochs = cfgs.train_ssl_epochs
        self.warmup_epochs = configs.warmup_epochs
        self.freeze_last_layer = configs.freeze_last_layer
        self.lr = configs.lr
        self.min_lr = configs.min_lr
        self.momentum = configs.momentum
        self.weight_decay = configs.weight_decay
        self.max_weight_decay = configs.max_weight_decay
        self.train_num_workers = configs.train_num_workers
        self.val_num_workers = configs.val_num_workers
        self.save_ckpt_folder = configs.save_ckpt_folder
        self.save_ckpt_interval = configs.save_ckpt_interval
        self.is_resume_model = configs.is_resume_model
        self.resume_path = configs.resume_path

        self.save_results_folder = configs.save_results_folder
        if not os.path.exists(os.path.join(self.save_results_folder)):
            os.makedirs(os.path.join(self.save_results_folder))

        if not os.path.exists(os.path.join(configs.log_path)):
            os.makedirs(os.path.join(configs.log_path))
        self.log_path = configs.log_path + f"/epoch{self.epochs}_lr{self.lr}_" \
                                           f"batchsize{configs.train_labeled_batch_size}_patchsize{self.patch_size}.txt"
        self.logger = Logger(os.path.join(self.log_path))

    def train(self):
        # set_seed(self.seed)

        labeled_data = HSICD_Dataset(dataset_name=self.dataset_name, mode='labeled', patch_size=cfgs.patch_size)
        # labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled_data)
        labeled_loader = DataLoader(labeled_data, batch_size=self.train_labeled_batch_size, shuffle=True,
                                    num_workers=self.train_num_workers, pin_memory=True)

        unlabeled_data = HSICD_Dataset(dataset_name=self.dataset_name, mode='unlabeled', patch_size=cfgs.patch_size)
        # unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_data)
        unlabeled_loader = DataLoader(unlabeled_data, batch_size=self.train_unlabeled_batch_size,
                                      shuffle=True, num_workers=self.train_num_workers, pin_memory=True)

        val_data = HSICD_Dataset(dataset_name=self.dataset_name, mode='val', patch_size=self.patch_size)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=self.val_batch_size, shuffle=False,
                                num_workers=self.val_num_workers, pin_memory=True)

        labeled_set = set(labeled_data.random_points)
        unlabeled_set = set(unlabeled_data.random_points)
        val_set = set(val_data.random_points)

        intersection = labeled_set.intersection(unlabeled_set)
        assert len(intersection) == 0, "The labeled set and unlabeled set have intersection!"
        intersection = labeled_set.intersection(val_set)
        assert len(intersection) == 0, "The labeled set and val set have intersection!"

        start_epoch = 0
        if self.is_resume_model:
            # if self.local_rank == 0:
            print('Load Checkpoint of Model...\n')
            start_epoch = self.resume_ckpt(self.resume_path)
            print(f"Resuming checkpoint of epoch {start_epoch}.")
            if (os.path.basename(self.resume_path).startswith('best')):
                    # os.path.basename(self.resume_path).startswith('checkpoint')):
                start_epoch = self.epochs
            # if self.local_rank == 0:
            print('Load Checkpoint of Model Successfully!\n')
            self.logger.logger.info(f"Resuming checkpoint of epoch {start_epoch}.")

        # if ddp.has_batchnorms(self.student):
        #     if self.local_rank == 0:
        #         print("Using SyncBatchNorm!")
        #     self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
        #     self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        #     self.classifier = nn.SyncBatchNorm.convert_sync_batchnorm(self.classifier)
        #     self.projhead_student = nn.SyncBatchNorm.convert_sync_batchnorm(self.projhead_student)
        #     self.projhead_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.projhead_teacher)
        # print("Using DistributedDataParallel!")
        # print(f"local_rank: {self.local_rank}, world_size: {self.world_size}")
        # self.student = nn.parallel.DistributedDataParallel(self.student, device_ids=[self.local_rank],
        #                                                    output_device=self.local_rank)
        # self.classifier = nn.parallel.DistributedDataParallel(self.classifier, device_ids=[self.local_rank],
        #                                                       output_device=self.local_rank)
        # self.projhead_student = nn.parallel.DistributedDataParallel(self.projhead_student,
        #                                                             device_ids=[self.local_rank],
        #                                                             output_device=self.local_rank)
        # self.teacher = nn.parallel.DistributedDataParallel(self.teacher, device_ids=[self.local_rank],
        #                                                    output_device=self.local_rank)
        # self.projhead_teacher = nn.parallel.DistributedDataParallel(self.projhead_teacher,
        #                                                             device_ids=[self.local_rank],
        #                                                             output_device=self.local_rank)
        #
        # self.optimizer = optim.SGD(list(self.student.parameters()) + list(self.projhead_student.parameters()) +
        #                            list(self.classifier.parameters()), lr=self.lr, momentum=self.momentum,
        #                            weight_decay=self.weight_decay)
        #
        # self.criterion_sl = FocalLoss().cuda()
        # self.criterion_ssl = ModifiedDINOLoss(self.out_dim, self.warmup_teacher_temp, self.teacher_temp,
        #                                       self.warmup_teacher_temp_epochs, self.train_ssl_epochs,
        #                                       gamma=self.gamma).cuda()

        # if self.local_rank == 0:
        print("****************Start Training****************\n")
        best_val_loss = float('inf')
        best_val_acc = 0.
        train_sl_losses = []
        train_ssl_losses = []
        train_losses = []
        val_losses = []
        for epoch in range(start_epoch, self.epochs):
            # labeled_sampler.set_epoch(epoch)
            # unlabeled_sampler.set_epoch(epoch)

            train_epoch_start_time = time.time()
            train_acc, train_sl_loss, train_ssl_loss, train_loss, batch_time = self.train_per_epoch(labeled_loader,
                                                                                                    unlabeled_loader,
                                                                                                    epoch)
            train_epoch_time = time.time() - train_epoch_start_time
            # self.scheduler.step()
            self.adjust_learning_rate(epoch)
            # self.adjust_weight_decay(epoch)
            self.adjust_momentum_teacher(epoch)

            val_epoch_start_time = time.time()
            val_acc, val_loss = self.val_per_epoch(val_loader)
            val_epoch_time = time.time() - val_epoch_start_time

            epoch_time = train_epoch_time + val_epoch_time

            eta_seconds = epoch_time * (self.epochs - epoch - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            # if self.local_rank == 0:
            train_sl_losses.append(train_sl_loss)
            train_ssl_losses.append(train_ssl_loss)
            train_losses.append(train_loss)
            with open(os.path.join(self.save_results_folder, 'train_loss.txt'), 'w') as f:
                for item in train_losses:
                    f.write(f"{item}\n")

            val_losses.append(val_loss)
            with open(os.path.join(self.save_results_folder, 'val_loss.txt'), 'w') as f:
                for item in val_losses:
                    f.write(f"{item}\n")

            print(f"Epoch: [{epoch + 1}/{self.epochs}] || lr: {self.get_lr()} || wd: {self.get_weight_decay()} ||"
                  f" Train_loss: {train_loss} || Train_sl_loss: {train_sl_loss} || Train_ssl_loss: {train_ssl_loss}"
                  f" || Train_acc: {(train_acc * 100):.2f}% || Val_loss: {val_loss} ||"
                  f" Val_acc: {(val_acc * 100):.2f}% || Batch_time: {batch_time:.4f}s || Epoch_time: {epoch_time}s"
                  f" || Train_epoch_time: {train_epoch_time:.4f}s || Val_epoch_time: {val_epoch_time:.4f}s ||"
                  f" ETA: {eta}")

            self.logger.get_train_logs(epoch, self.epochs, self.get_lr(), train_loss, train_acc, val_loss, val_acc,
                                       batch_time, epoch_time, train_epoch_time, val_epoch_time, eta)

            if (epoch + 1) % self.save_ckpt_interval == 0:
                self.sava_ckpt(self.save_ckpt_folder, epoch, 'checkpoint')

            if val_loss < best_val_loss and val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                self.sava_ckpt(self.save_ckpt_folder, epoch, 'best')
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                self.sava_ckpt(self.save_ckpt_folder, epoch, 'best_loss')
            elif val_acc > best_val_acc:
                best_val_acc = val_acc
                self.sava_ckpt(self.save_ckpt_folder, epoch, 'best_acc')

        if start_epoch < self.epochs:
            self.plot_loss_curve(train_sl_losses, train_ssl_losses, train_losses, val_losses, self.save_results_folder)

    def train_per_epoch(self, labeled_loader, unlabeled_loader, epoch):
        self.student.train()
        self.teacher.train()
        self.classifier.train()
        self.projhead_student.train()
        self.projhead_teacher.train()

        data_loader_len = len(labeled_loader)
        if epoch < self.train_sl_epochs:
            data_loader = labeled_loader
        else:
            data_loader = zip(labeled_loader, unlabeled_loader)

        total_loss = 0
        total_sl_loss = 0
        total_ssl_loss = 0
        ssl_loss = 0
        correct = 0
        total_num = 0
        batch_total_time = 0
        student_param_norms = None
        projhead_student_norms = None
        classifier_param_norms = None
        for batch_idx, data in enumerate(data_loader):
            batch_start_time = time.time()

            if epoch < self.train_sl_epochs:
                img_t1, img_t2, label, _ = data

                img_t1 = img_t1.cuda()
                img_t2 = img_t2.cuda()
                label = label.cuda()

                logits_sl = self.classifier(self.student(img_t1), self.student(img_t2))
                sl_loss = self.criterion_sl(logits_sl, label.long())
                self.optimizer.zero_grad()
                sl_loss.backward()
                if self.use_clip_grad:
                    student_param_norms = self.clip_gradients(self.student, self.clip_grad)
                    classifier_param_norms = self.clip_gradients(self.classifier, self.clip_grad)
                self.optimizer.step()

            else:
                if self.ssl_start:
                    self.teacher.load_state_dict(self.student.state_dict())
                    for p in self.teacher.parameters():
                        p.requires_grad = False
                    self.projhead_teacher.load_state_dict(self.projhead_student.state_dict())
                    for p in self.projhead_teacher.parameters():
                        p.requires_grad = False
                    self.ssl_start = False

                (img_t1, img_t2, label, _), (img_t1_u, img_t2_u, _, _) = data

                img_t1 = img_t1.cuda()
                img_t2 = img_t2.cuda()
                label = label.cuda()
                img_t1_u = img_t1_u.cuda()
                img_t2_u = img_t2_u.cuda()

                logits_sl = self.classifier(self.student(img_t1), self.student(img_t2))
                sl_loss = self.criterion_sl(logits_sl, label.long())

                student_sf_t1 = self.student(img_t1_u)
                student_sf_t2 = self.student(img_t2_u)
                student_diff = torch.abs(student_sf_t1 - student_sf_t2)
                student_output_t1 = self.projhead_student(student_sf_t1)
                student_output_t2 = self.projhead_student(student_sf_t2)
                student_output_diff = self.projhead_student(student_diff)
                teacher_sf_t1 = self.teacher(img_t1_u)
                teacher_sf_t2 = self.teacher(img_t2_u)
                teacher_diff = torch.abs(teacher_sf_t1 - teacher_sf_t2)
                teacher_output_t1 = self.projhead_teacher(teacher_sf_t1)
                teacher_output_t2 = self.projhead_teacher(teacher_sf_t2)
                teacher_output_diff = self.projhead_teacher(teacher_diff)
                ssl_loss = self.criterion_ssl(student_output_t1, student_output_t2, student_output_diff,
                                              teacher_output_t1, teacher_output_t2, teacher_output_diff,
                                              epoch - self.train_sl_epochs)

                loss = sl_loss + self.eta * ssl_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_clip_grad:
                    student_param_norms = self.clip_gradients(self.student, self.clip_grad)
                    projhead_student_norms = self.clip_gradients(self.projhead_student, self.clip_grad)
                    classifier_param_norms = self.clip_gradients(self.classifier, self.clip_grad)
                self.cancel_gradients_last_layer(epoch, self.projhead_student, self.freeze_last_layer)
                self.optimizer.step()

                # EMA update for the teacher
                with torch.no_grad():
                    for param_teacher, param_student in zip(self.teacher.parameters(),
                                                            self.student.parameters()):
                        param_teacher.data.mul_(self.momentum_teacher).add_(param_student.detach().data,
                                                                            alpha=1 - self.momentum_teacher)
                    for param_projhead_teacher,\
                            param_projhead_student in zip(self.projhead_teacher.parameters(),
                                                          self.projhead_student.parameters()):
                        param_projhead_teacher.data.mul_(self.momentum_teacher).add_(
                            param_projhead_student.detach().data,
                            alpha=1 - self.momentum_teacher)

            # torch.cuda.synchronize()

            batch_total_time += time.time() - batch_start_time

            preds = torch.argmax(logits_sl, dim=1)
            correct += (preds == label).sum().item()
            total_num += label.size(0)

            total_sl_loss += sl_loss.item()
            if epoch < self.train_sl_epochs:
                total_ssl_loss = 0
                total_loss += sl_loss.item()
            else:
                total_ssl_loss += ssl_loss.item()
                total_loss += loss.item()

        batch_time = 0
        train_acc = correct / total_num
        train_sl_loss = total_sl_loss / data_loader_len
        train_ssl_loss = total_ssl_loss / data_loader_len
        train_loss = total_loss / data_loader_len

        # train_acc, train_sl_loss = ddp.calculate_global_accuracy_and_loss(correct, total_num,
        #                                                                   total_sl_loss / data_loader_len,
        #                                                                   dist.get_world_size())
        # _, train_ssl_loss = ddp.calculate_global_accuracy_and_loss(0, 0, total_ssl_loss / data_loader_len,
        #                                                            dist.get_world_size())
        # _, train_loss = ddp.calculate_global_accuracy_and_loss(0, 0, total_loss / data_loader_len,
        #                                                        dist.get_world_size())
        # if self.local_rank == 0:
        batch_time = batch_total_time / data_loader_len
        if projhead_student_norms:
            print(f"\nmax_student_param_norms: {max(student_param_norms)},"
                  f" max_projhead_student_norms: {max(projhead_student_norms)},"
                  f" max_classifier_param_norms: {max(classifier_param_norms)}")
        else:
            print(f"\nmax_student_param_norms: {max(student_param_norms)},"
                  f" max_classifier_param_norms: {max(classifier_param_norms)}")

        return train_acc, train_sl_loss, train_ssl_loss, train_loss, batch_time

    def val_per_epoch(self, data_loader):
        self.student.eval()
        self.classifier.eval()
        total_loss = 0
        correct = 0
        total_num = 0

        with torch.no_grad():
            for batch_idx, (img_t1, img_t2, label, _) in enumerate(data_loader):
                img_t1 = img_t1.cuda()
                img_t2 = img_t2.cuda()
                label = label.cuda()

                logits = self.classifier(self.student(img_t1), self.student(img_t2))
                loss = self.criterion_sl(logits, label.long())

                # torch.cuda.synchronize()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()
                total_num += label.size(0)

                total_loss += loss.item()

            # if self.local_rank == 0:
            # val_acc, val_loss = ddp.calculate_global_accuracy_and_loss(correct, total_num,
            #                                                            total_loss / len(data_loader),
            #                                                            dist.get_world_size())
            val_acc = correct / total_num
            val_loss = total_loss / len(data_loader)

        return val_acc, val_loss

    def sava_ckpt(self, save_path, epoch, name):
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(save_path)

        if name == 'checkpoint':
            ckpt_path = os.path.join(save_path, f"{name}_epoch{epoch + 1}.pth")
        elif name == 'best' or name == 'best_loss' or name == 'best_acc':
            ckpt_path = os.path.join(save_path, f"{name}.pth")
        else:
            raise ValueError(f"Invalid name for saving ckpt. Expected one of: checkpoint, best.")

        torch.save({
            'epoch': epoch + 1,
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'projhead_student_state_dict': self.projhead_student.state_dict(),
            'projhead_teacher_state_dict': self.projhead_teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
        }, ckpt_path)

        print(f"Model saved at {ckpt_path}!")

    def resume_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        epoch = checkpoint['epoch']
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.projhead_student.load_state_dict(checkpoint['projhead_student_state_dict'])
        self.projhead_teacher.load_state_dict(checkpoint['projhead_teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return epoch

    def plot_loss_curve(self, train_sl_losses, train_ssl_losses, train_losses, val_losses, save_path):
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(train_sl_losses, label='Training SL Loss')
        plt.plot(train_ssl_losses, label='Training SSL Loss')
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
        plt.show()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_weight_decay(self):
        for param_group in self.optimizer.param_groups:
            return param_group['weight_decay']

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                    1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_weight_decay(self, epoch):
        if epoch < self.warmup_epochs:
            weight_decay = self.weight_decay * epoch / self.warmup_epochs
        else:
            weight_decay = self.max_weight_decay + (self.weight_decay - self.max_weight_decay) * 0.5 * (
                    1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = weight_decay

    def adjust_momentum_teacher(self, epoch):
        if epoch < self.train_sl_epochs:
            self.momentum_teacher = self.momentum_teacher
        else:
            self.momentum_teacher = 1. + (self.momentum_teacher - 1) * 0.5 * (
                    1. + math.cos(math.pi * (epoch - self.train_sl_epochs) / self.train_ssl_epochs))

    def cancel_gradients_last_layer(self, epoch, model, freeze_last_layer):
        if epoch >= self.train_sl_epochs + freeze_last_layer:
            return
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def clip_gradients(self, model, clip):
        norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
        return norms


if __name__ == '__main__':
    model = 0
    criterion = 0
    optimizer = 0
    scheduler = 0
    # my_trainer = Trainer(model, criterion, optimizer, scheduler, cfgs)
    x = 1 + (0.996 - 1) * 0.5 * (1. + math.cos(math.pi * 75. / 150.))
    print(x)
