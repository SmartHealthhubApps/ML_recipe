import torch


img_path = 'dataset/images'
lbl_path = 'dataset/labels'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
