import torch


class CFG:
    img_path = 'dataset_pascal-voc/images'
    ann_path = 'dataset_pascal-voc/annotations'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    max_len = 1000
    img_size = 384
    num_bins = img_size
    pad_idx = None  # Should be filled

    batch_size = 16
    epochs = 10

    model_name = 'deit3_small_patch16_384_in21ft1k'
    num_patches = 576
    lr = 1e-4
    weight_decay = 1e-4

    generation_steps = 101


print(f'Using device: {CFG.device}')
