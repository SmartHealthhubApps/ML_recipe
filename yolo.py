#necessary libs
import torch
from ultralytics import YOLO

# fine-tuning and train
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading models choose a particular version of YOLO
    model = YOLO(model="/home/aamotyakin/yolov10x.pt").to(device)

    # # Freezing weigths
    # freeze = [f"model.{x}." for x in range(23)]
    # for k, v in model.named_parameters():
    #     v.requires_grad = True
    #     if any(x in k for x in freeze):
    #         print(f"Freezing {k}")
    #         v.requires_grad = False

    # Path to dataset
    dataset_path = "/home/aamotyakin/Dataset_wo_lime_broc_peas_spin_copy/TRAIN.yaml"
    # # Getting augmentations
    # transform = get_transform()

    # Training
    model.train(data=dataset_path,
                imgsz=640,
                epochs=200,
                batch=64,
                name='yolov10_transfer_learning',
                device=[0, 1, 2, 3])

if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    train()
