import os
import random
import numpy as np
from utils.seed import set_seed
from dataloader.get_datasets import get_dataset


def points_generation(dataset_name='China', train_ratio=0.2, val_ratio=0.3, label_ratio=0.01):
    set_seed(42)

    img_t1, img_t2, img_gt = get_dataset(dataset_name)
    print(img_t1.shape, img_t2.shape, img_gt.shape)
    if dataset_name == 'BayArea' or dataset_name == 'Barbara':
        img_gt_tmp = np.zeros_like(img_gt)
        img_gt_tmp[img_gt == 1.] = 2.
        img_gt[img_gt == 2.] = 1.
        img_gt[img_gt_tmp == 2.] = 2.
    h, w = img_gt.shape[0], img_gt.shape[1]
    all_num = h * w
    print(f"all_num:{all_num}")

    whole_point = img_gt.reshape(1, all_num)

    if dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River':
        changed_indices = np.where(whole_point[0] == 1.)[0].tolist()
        unchanged_indices = np.where(whole_point[0] == 0.)[0].tolist()
    elif dataset_name == 'BayArea' or dataset_name == 'Barbara':
        changed_indices = np.where(whole_point[0] == 2.)[0].tolist()
        unchanged_indices = np.where(whole_point[0] == 1.)[0].tolist()
    print(f"changed_indices:{len(changed_indices)}, unchanged_indices:{len(unchanged_indices)}")

    num_changed_train = int(len(changed_indices) * train_ratio)
    num_unchanged_train = int(len(unchanged_indices) * train_ratio)
    num_changed_labeled = int(len(changed_indices) * label_ratio)
    num_unchanged_labeled = int(len(unchanged_indices) * label_ratio)
    num_changed_val = int(len(changed_indices) * val_ratio)
    num_unchanged_val = int(len(unchanged_indices) * val_ratio)


    changed_points_train = random.sample(changed_indices, num_changed_train)
    unchanged_points_train = random.sample(unchanged_indices, num_unchanged_train)
    train_points = changed_points_train + unchanged_points_train

    changed_points_labeled = random.sample(changed_points_train, num_changed_labeled)
    unchanged_points_labeled = random.sample(unchanged_points_train, num_unchanged_labeled)
    labeled_points = changed_points_labeled + unchanged_points_labeled

    changed_points_unlabeled = list(set(changed_points_train) - set(changed_points_labeled))
    unchanged_points_unlabeled = list(set(unchanged_points_train) - set(unchanged_points_labeled))
    unlabeled_points = changed_points_unlabeled + unchanged_points_unlabeled

    changed_points_val = random.sample(list(set(changed_indices) - set(changed_points_train)), num_changed_val)
    unchanged_points_val = random.sample(list(set(unchanged_indices) - set(unchanged_points_train)), num_unchanged_val)
    val_points = changed_points_val + unchanged_points_val

    print(f"train_points:{len(train_points)}, labeled_points:{len(labeled_points)},"
          f" unlabeled_points:{len(unlabeled_points)}, val_points:{len(val_points)}")

    print(f"labeled_points:{labeled_points[:10]}")
    print(f"unlabeled_points:{unlabeled_points[:10]}")
    print(f"val_points:{val_points[:10]}")

    labeled_set = set(labeled_points)
    unlabeled_set = set(unlabeled_points)
    val_set = set(val_points)

    intersection = labeled_set.intersection(unlabeled_set)
    assert len(intersection) == 0, "The labeled set and unlabeled set have intersection!"
    intersection = labeled_set.intersection(val_set)
    assert len(intersection) == 0, "The labeled set and val set have intersection!"

    return img_gt, train_points, labeled_points, unlabeled_points, val_points


def save_points_to_file(points, dataset_name, mode):
    file_name = os.path.join(f"../datasets/{dataset_name}", f"{mode}_points.txt")

    with open(file_name, 'w') as file:
        for point in points:
            file.write(f"{point}\n")


if __name__ == '__main__':
    dataset_name = 'Barbara'
    train_ratio = 0.2
    val_ratio = 0.3
    label_ratio = 0.01

    img_gt, train_points, labeled_points, unlabeled_points, val_points = points_generation(dataset_name, train_ratio,
                                                                                           val_ratio, label_ratio)
    save_points_to_file(train_points, dataset_name, 'train')
    save_points_to_file(labeled_points, dataset_name, 'labeled')
    save_points_to_file(unlabeled_points, dataset_name, 'unlabeled')
    save_points_to_file(val_points, dataset_name, 'val')
