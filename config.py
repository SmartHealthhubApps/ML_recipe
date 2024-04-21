import torch


class CFG:
    img_path = 'dataset_pascal-voc/images'
    ann_path = 'dataset_pascal-voc/annotations'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.devcice('mps')
    else:
        device = torch.device('cpu')

    max_len = 300
    img_size = 640
    num_bins = img_size

    batch_size = 16
    epochs = 10

    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 1600
    lr = 1e-4
    weight_decay = 1e-4

    generation_steps = 101
