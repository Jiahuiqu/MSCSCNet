import os


class Configs:
    model_name = 'MSCSC'
    device = '0'
    random_seed = 42

    """**************** dataset and directory ****************"""
    dataset_name = 'China'  # China, Hermiston, River, BayArea, Barbara
    patch_size = 11

    """**************** Hyper Arguments for Training ****************"""
    in_feats = 154
    hi_feats = 256
    iter_num = 3
    out_dim = 256
    warmup_teacher_temp = 0.04
    teacher_temp = 0.07
    warmup_teacher_temp_epochs = 8
    eta = 0.7
    gamma = 1.6
    momentum_teacher = 0.917  # China: 0.917, River: 0.918, Barbara: 0.917
    clip_grad = 0.3
    use_clip_grad = True

    train_labeled_batch_size = 32
    train_unlabeled_batch_size = 512
    val_batch_size = 1000
    epochs = 100
    train_sl_epochs = 15
    train_ssl_epochs = 85
    warmup_epochs = 5
    freeze_last_layer = 0
    train_num_workers = 0
    val_num_workers = 8
    save_ckpt_interval = 100
    is_resume_model = False

    """**************** Hyper Arguments for Testing ****************"""
    test_batch_size = 1000
    test_num_workers = 8

    """**************** Hyper Arguments for Optimizer ****************"""
    lr = 32e-3
    min_lr = 32e-6
    weight_decay = 5e-3  # for SGD and Adam
    max_weight_decay = 2e-2
    eps = 1e-4
    momentum = 0.9  # for SGD
    # step scheduler
    lr_step = 50  # step
    milestones = [50, 100, 150]  # multistep
    lr_gamma = 0.1

    """**************** Log and Save Folder Path ****************"""
    log_path = 'logs/' + model_name + '_' + dataset_name
    save_ckpt_folder = 'checkpoints/' + model_name + '_' + dataset_name + f"/epoch{epochs}_lr{lr}_" \
                                                                          f"batchsize{train_labeled_batch_size}_" \
                                                                          f"patchsize{patch_size}"
    resume_path = 'checkpoints/' + model_name + '_' + dataset_name + f'/epoch{epochs}_lr{lr}_' \
                                                                     f'batchsize{train_labeled_batch_size}_' \
                                                                     f'patchsize{patch_size}/best_acc.pth'
    save_results_folder = 'results/' + model_name + '_' + dataset_name + f"/epoch{epochs}_lr{lr}_" \
                                                                         f"batchsize{train_labeled_batch_size}_" \
                                                                         f"patchsize{patch_size}"


if __name__ == '__main__':
    # if not os.path.exists(os.path.join(os.getcwd(), Configs.save_folder)):
    #     os.makedirs(Configs.save_folder)
    print(os.path.exists(os.path.join(Configs.save_ckpt_folder)))
