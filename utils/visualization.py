import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat


def predict2img(predict, img_gt, pos, dataset_name, save_folder):
    if not os.path.exists(os.path.join(save_folder)):
        os.makedirs(os.path.join(save_folder))

    predict_img = np.zeros_like(img_gt)
    if dataset_name == 'China' or dataset_name == 'Hermiston' or dataset_name == 'River':
        predict_img = np.zeros_like(img_gt)
    elif dataset_name == 'BayArea' or dataset_name == 'Barbara':
        predict_img = np.full_like(img_gt, 0.5)

    for i in range(len(predict)):
        x = pos[i][0]
        y = pos[i][1]
        v = predict[i]

        predict_img[x][y] = v

    savemat(os.path.join(save_folder, 'predict_mat.mat'), {'pred': predict_img})
    print("Predict_mat has been saved!")

    plt.imshow(predict_img, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imsave(os.path.join(save_folder, 'predict_img.png'), predict_img, cmap='gray')

    print("Predict_img has been saved!")


if __name__ == '__main__':
    pass
