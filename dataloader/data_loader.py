import torch
import numpy as np
import cv2
import os
import random
from dataloader.get_datasets import get_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from utils.seed import set_seed
from mmengine.registry import DATASETS


# class HSICD_Dataset(Dataset):
#     def __init__(self, dataset_name='China', mode='train', patch_size=9, train_val_ratio=0.2, label_ratio=0.01):
#         super().__init__()
#
#         self.dataset_name = dataset_name
#         self.mode = mode
#         self.padding = int(patch_size / 2)
#         self.train_val_ratio = train_val_ratio
#         self.label_ratio = label_ratio
#
#         # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
#         if self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River':
#             self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name)
#             self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
#             self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
#             self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
#                                                      self.padding,
#                                                      cv2.BORDER_REPLICATE)
#             self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
#                                                      self.padding,
#                                                      cv2.BORDER_REPLICATE)
#             self.random_points = self.get_random_points(self.img_gt, self.dataset_name)
#
#         # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
#         elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara':
#             self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name)
#             self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
#             img_gt_tmp = np.zeros_like(self.img_gt)
#             img_gt_tmp[self.img_gt == 1.] = 2.
#             self.img_gt[self.img_gt == 2.] = 1.
#             self.img_gt[img_gt_tmp == 2.] = 2.
#
#             self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
#             self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
#                                                      self.padding,
#                                                      cv2.BORDER_REPLICATE)
#             self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
#                                                      self.padding,
#                                                      cv2.BORDER_REPLICATE)
#             self.random_points = self.get_random_points(self.img_gt, self.dataset_name)
#
#         else:
#             raise ValueError(f"Invalid dataset name {dataset_name}. Expected one of: China, Hermiston, River, "
#                              f"BayArea, Barbara.")
#
#     def get_random_points(self, gt, dataset_name):
#         all_num = self.h * self.w
#         whole_point = gt.reshape(1, all_num)
#         random_points = []
#
#         changed_indices = []
#         unchanged_indices = []
#         if dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River':
#             changed_indices = np.where(whole_point[0] == 1.)[0].tolist()
#             unchanged_indices = np.where(whole_point[0] == 0.)[0].tolist()
#         elif dataset_name == 'BayArea' or dataset_name == 'Barbara':
#             changed_indices = np.where(whole_point[0] == 2.)[0].tolist()
#             unchanged_indices = np.where(whole_point[0] == 1.)[0].tolist()
#
#         num_changed_train = int(len(changed_indices) * self.train_val_ratio)
#         num_unchanged_train = int(len(unchanged_indices) * self.train_val_ratio)
#         num_changed_labeled = int(len(changed_indices) * self.label_ratio)
#         num_unchanged_labeled = int(len(unchanged_indices) * self.label_ratio)
#
#         changed_points_train = random.sample(changed_indices, num_changed_train)
#         unchanged_points_train = random.sample(unchanged_indices, num_unchanged_train)
#
#         changed_points_labeled = random.sample(changed_points_train, num_changed_labeled)
#         unchanged_points_labeled = random.sample(unchanged_points_train, num_unchanged_labeled)
#
#         changed_points_unlabeled = list(set(changed_points_train) - set(changed_points_labeled))
#         unchanged_points_unlabeled = list(set(unchanged_points_train) - set(unchanged_points_labeled))
#
#         changed_points_val = list(set(changed_indices) - set(changed_points_train))
#         unchanged_points_val = list(set(unchanged_indices) - set(unchanged_points_train))
#
#         if self.mode == 'train':
#             random_points = changed_points_train + unchanged_points_train
#         elif self.mode == 'val':
#             random_points = changed_points_val + unchanged_points_val
#         elif self.mode == 'test':
#             if dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River':
#                 random_points = list(range(all_num))
#             elif dataset_name == 'BayArea' or dataset_name == 'Barbara':
#                 random_points = np.nonzero(whole_point[0])[0].tolist()
#         elif self.mode == 'labeled':
#             random_points = changed_points_labeled + unchanged_points_labeled
#         elif self.mode == 'unlabeled':
#             random_points = changed_points_unlabeled + unchanged_points_unlabeled
#         else:
#             raise ValueError(f"Invalid mode {self.mode}. Expected one of: train, val, test, labeled, unlabeled.")
#
#         return random_points
#
#     def data_preprocess(self, img):
#         mean = img.mean(axis=(0, 1))
#         std = img.std(axis=(0, 1))
#
#         img_normalized = (img - mean) / std
#
#         return img_normalized
#
#     def __len__(self):
#         return len(self.random_points)
#
#     def __getitem__(self, index):
#         original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
#         new_i = original_i + self.padding
#         new_j = original_j + self.padding
#
#         img_patch_t1 = self.img_t1_padding[new_i - self.padding:new_i + self.padding + 1,
#                        new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
#         img_patch_t2 = self.img_t2_padding[new_i - self.padding:new_i + self.padding + 1,
#                        new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
#
#         img_patch_t1 = torch.from_numpy(img_patch_t1)
#         img_patch_t2 = torch.from_numpy(img_patch_t2)
#
#         if self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River':
#             label = self.img_gt[original_i, original_j]
#         elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara':
#             label = self.img_gt[original_i, original_j] - 1.
#         label = torch.tensor(label, dtype=torch.float32)
#
#         return img_patch_t1, img_patch_t2, label, torch.tensor((original_i, original_j), dtype=torch.long)


class HSICD_Dataset(Dataset):
    def __init__(self, dataset_name='China', mode='train', patch_size=9):
        super().__init__()

        self.dataset_name = dataset_name
        self.mode = mode
        self.padding = int(patch_size / 2)

        # 全标数据集 0为unchanged 1为changed 其中对于River数据集，255为changed，已经将其gt除以255
        if self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River':
            self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name)
            self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.random_points = self.read_points()

        # 未全标数据集 1为changed 2为unchanged 所以得进行调整，将gt中的1变为2,2变为1
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara':
            self.img_t1, self.img_t2, self.img_gt = get_dataset(self.dataset_name)
            self.img_t1, self.img_t2 = self.data_preprocess(self.img_t1), self.data_preprocess(self.img_t2)
            img_gt_tmp = np.zeros_like(self.img_gt)
            img_gt_tmp[self.img_gt == 1.] = 2.
            self.img_gt[self.img_gt == 2.] = 1.
            self.img_gt[img_gt_tmp == 2.] = 2.

            self.h, self.w = self.img_gt.shape[0], self.img_gt.shape[1]
            self.img_t1_padding = cv2.copyMakeBorder(self.img_t1, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.img_t2_padding = cv2.copyMakeBorder(self.img_t2, self.padding, self.padding, self.padding,
                                                     self.padding,
                                                     cv2.BORDER_REFLECT)
            self.random_points = self.read_points()

        else:
            raise ValueError(f"Invalid dataset name {dataset_name}. Expected one of: China, Hermiston, River, "
                             f"BayArea, Barbara.")

    def read_points(self):
        points = []

        if self.mode != 'test':
            file_name = os.path.join(f"datasets/{self.dataset_name}", f"{self.mode}_points.txt")

            with open(file_name, 'r') as file:
                for line in file:
                    point = int(line.strip())
                    points.append(point)
        else:
            all_num = self.h * self.w
            if self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River':
                points = list(range(all_num))
            elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara':
                whole_point = self.img_gt.reshape(1, all_num)
                points = np.nonzero(whole_point[0])[0].tolist()

        return points

    def data_preprocess(self, img):
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))

        img_normalized = (img - mean) / std

        return img_normalized

    def __len__(self):
        return len(self.random_points)

    def __getitem__(self, index):
        original_i, original_j = divmod(self.random_points[index], self.w)  # 第几行，第几列
        new_i = original_i + self.padding
        new_j = original_j + self.padding

        img_patch_t1 = self.img_t1_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
        img_patch_t2 = self.img_t2_padding[new_i - self.padding:new_i + self.padding + 1,
                       new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)

        img_patch_t1 = torch.from_numpy(img_patch_t1)
        img_patch_t2 = torch.from_numpy(img_patch_t2)

        if self.dataset_name == 'China' or self.dataset_name == 'Hermiston' or self.dataset_name == 'River':
            label = self.img_gt[original_i, original_j]
        elif self.dataset_name == 'BayArea' or self.dataset_name == 'Barbara':
            label = self.img_gt[original_i, original_j] - 1.
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return img_patch_t1, img_patch_t2, label, torch.tensor((original_i, original_j), dtype=torch.long)


if __name__ == '__main__':
    random.seed(123)
    data_list = [i for i in range(100)]
    sampled_data = random.sample(data_list, 49)
    print(sampled_data)
    train_data = HSICD_Dataset(mode='train')
    labeled_data = HSICD_Dataset(mode='labeled')
    unlabeled_data = HSICD_Dataset(mode='unlabeled')
    val_data = HSICD_Dataset(mode='val')
    img_t1, img_t2, label, pos= train_data.__getitem__(1)
    print(label.dtype)
    # print(train_data.random_points[:10])
    # print(labeled_data.random_points[:10])
    # print(unlabeled_data.random_points[:10])
    # print(val_data.random_points[:10])
    # train_loader = DataLoader(dataset, 32, shuffle=True)
    # img1, img2, label, pos = next(iter(train_loader))
    # print(img1)
    # print(img2.shape)
    # print(label)
    # print(pos.shape)
