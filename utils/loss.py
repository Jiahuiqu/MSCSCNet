import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, label):
        cross_entropy_loss = F.cross_entropy(logits, label, reduction='none')
        pt = torch.exp(-cross_entropy_loss)
        alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * cross_entropy_loss

        return focal_loss.mean()


class ModifiedDINOLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, gamma=2.0):
        super().__init__()
        self.student_temp = torch.tensor(student_temp)
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("diff_center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.gamma = gamma

    def forward(self, student_output_t1, student_output_t2, student_output_diff, teacher_output_t1, teacher_output_t2,
                teacher_output_diff, epoch):
        temp = self.teacher_temp_schedule[epoch]
        teacher_output_t1 = teacher_output_t1.detach()
        teacher_output_t2 = teacher_output_t2.detach()
        teacher_output_diff = teacher_output_diff.detach()

        # Calculate the self-supervised loss for static scenarios
        L_u = -0.5 * (F.log_softmax(student_output_t1 / self.student_temp, dim=-1) * F.softmax(
            (teacher_output_t2 - self.center) / temp, dim=-1) +
                      F.log_softmax(student_output_t2 / self.student_temp, dim=-1) * F.softmax(
                    (teacher_output_t1 - self.center) / temp, dim=-1)).sum(dim=-1)

        # Calculate the self-supervised loss for dynamic scenarios
        L_d = -torch.sum(
            F.log_softmax(student_output_diff / self.student_temp, dim=-1) * F.softmax(
                (teacher_output_diff - self.diff_center) / temp, dim=-1), dim=-1)

        # Calculate dynamic weight
        sim = F.cosine_similarity(teacher_output_t1 - self.center, teacher_output_t2 - self.center, dim=-1)
        psi = ((1 + sim) / 2) ** self.gamma

        # Calculate overall self-supervised loss
        L_ssl = (psi * L_u + (1 - psi) * L_d).mean()

        # Update the center
        self.update_center(torch.cat((teacher_output_t1, teacher_output_t2), dim=0))
        self.update_diff_center(teacher_output_diff)

        return L_ssl

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher_output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.barrier()
        # dist.all_reduce(batch_center)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_diff_center(self, teacher_output_diff):
        """
        Update center used for teacher_output_diff.
        """
        diff_batch_center = torch.sum(teacher_output_diff, dim=0, keepdim=True)
        # dist.barrier()
        # dist.all_reduce(diff_batch_center)
        diff_batch_center = diff_batch_center / len(teacher_output_diff)
        self.diff_center = self.diff_center * self.center_momentum + diff_batch_center * (1 - self.center_momentum)


if __name__ == '__main__':
    teacher_temp_schedule = np.concatenate((np.linspace(0.04, 0.07, 7), np.ones(70-7) * 0.07))
    print(teacher_temp_schedule)
    print(teacher_temp_schedule.shape)
    print(teacher_temp_schedule[84 - 15])
    # s1 = torch.randn(1024, 512)
    # s2 = torch.randn(1024, 512)
    # t1 = torch.randn(1024, 512)
    # t2 = torch.randn(1024, 512)
    # loss = -(F.log_softmax(s1 / 0.04, dim=-1) * F.softmax(s1 / 0.04, dim=-1)).sum(dim=-1)
    # loss_f = F.cross_entropy(s1 / 0.1, t2 / 0.04, reduction='none')
    # print(loss)
    # print(loss_f)
    # print(loss.mean())
    # criterion = ModifiedDINOLoss(65536, 0.1, 0.1, 0, 100)
    # loss = criterion(s1, s2, t1, t2, 0)
    # print(loss)
    # diff_student = torch.abs(s1 - s2)
    # diff_teacher = torch.abs(t1 - t2)
    #
    # L_u = -0.5 * (F.log_softmax(s1 / 0.1, dim=-1) * F.softmax(t2 / 0.04, dim=-1) +
    #               F.log_softmax(s2 / 0.1, dim=-1) * F.softmax(t1 / 0.04, dim=-1)).sum(dim=-1)
    # L_d = -torch.sum(F.log_softmax(diff_student / 0.1, dim=-1) * F.softmax(diff_teacher / 0.1, dim=-1), dim=-1)
    # sim = F.cosine_similarity(t1, t2, dim=-1)
    # psi = ((1 + sim) / 2) ** 2
    # L_ssl = (psi * L_u + (1 - psi) * L_d).mean()
    # print(L_ssl)
