import os
from scipy.io import loadmat


def get_China_dataset():
    """
    420 × 140 × 154, 40417 unchanged, 18383 changed
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'datasets/China/China_Change_Dataset.mat')

    data = loadmat(data_path)
    img_t1 = data['T1'].astype('float32')
    img_t2 = data['T2'].astype('float32')
    img_gt = data['Binary'].astype('float32')

    return img_t1, img_t2, img_gt


def get_Hermiston_dataset():
    """
    307 × 241 × 154, 57311 unchanged, 16676 changed
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'datasets/Hermiston-USA/USA_Change_Dataset.mat')

    data = loadmat(data_path)
    img_t1 = data['T1'].astype('float32')
    img_t2 = data['T2'].astype('float32')
    img_gt = data['Binary'].astype('float32')

    return img_t1, img_t2, img_gt


def get_River_dataset():
    """
    463 × 241 × 198, 101885 unchanged, 9698 changed
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    data1_path = os.path.join(current_path, 'datasets/River/river_before.mat')
    data2_path = os.path.join(current_path, 'datasets/River/river_after.mat')
    data3_path = os.path.join(current_path, 'datasets/River/groundtruth.mat')

    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    img_t1 = data_t1['river_before'].astype('float32')
    img_t2 = data_t2['river_after'].astype('float32')
    img_gt = (data_gt['lakelabel_v1'] / 255).astype('float32')

    return img_t1, img_t2, img_gt


def get_BayArea_dataset():
    """
    600 × 500 × 224, 34211 unchanged, 39270 changed, 226519 undetermined
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    data1_path = os.path.join(current_path, 'datasets/BayArea/Bay_Area_2013.mat')
    data2_path = os.path.join(current_path, 'datasets/BayArea/Bay_Area_2015.mat')
    data3_path = os.path.join(current_path, 'datasets/BayArea/bayArea_gtChanges2.mat')

    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    img_t1 = data_t1['HypeRvieW'].astype('float32')
    img_t2 = data_t2['HypeRvieW'].astype('float32')
    img_gt = data_gt['HypeRvieW'].astype('float32')

    return img_t1, img_t2, img_gt


def get_Barbara_dataset():
    """
    984 × 740 × 224, 80418 unchanged, 52134 changed, 595608 undetermined
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    data1_path = os.path.join(current_path, 'datasets/Barbara/barbara_2013.mat')
    data2_path = os.path.join(current_path, 'datasets/Barbara/barbara_2014.mat')
    data3_path = os.path.join(current_path, 'datasets/Barbara/barbara_gtChanges.mat')

    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    img_t1 = data_t1['HypeRvieW'].astype('float32')
    img_t2 = data_t2['HypeRvieW'].astype('float32')
    img_gt = data_gt['HypeRvieW'].astype('float32')

    return img_t1, img_t2, img_gt


def get_dataset(dataset_name):
    if dataset_name == 'China':
        return get_China_dataset()
    elif dataset_name == 'Hermiston':
        return get_Hermiston_dataset()
    elif dataset_name == 'River':
        return get_River_dataset()
    elif dataset_name == 'BayArea':
        return get_BayArea_dataset()
    elif dataset_name == 'Barbara':
        return get_Barbara_dataset()


if __name__ == '__main__':
    img_t1, img_t2, img_gt = get_dataset('Barbara')
    print(img_t1.shape)
    import torch
    from torchvision import transforms
    img_t1 = torch.from_numpy(img_t1.transpose(2, 0, 1))
    img_t1 = img_t1
    print(img_t1.shape)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
    ])
    img_t1 = transform(img_t1)
    print(img_t1.shape)
    img_t1 = img_t1.cpu().numpy()
    print(img_t1.shape)

    from scipy.io import savemat

    savemat('barbara.mat', {'HypeRvieW': img_t1.transpose(1, 2, 0)})
