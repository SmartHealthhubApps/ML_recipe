import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import cv2
from config import CFG


# Parsing XML
class XMLParser:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self._root = ET.parse(self.xml_file).getroot()
        self._objects = self._root.findall("object")
        self.img_path = os.path.join(CFG.img_path, self._root.find('filename').text)
        self.image_id = self._root.find("filename").text
        self.names = self._get_names()
        self.boxes = self._get_bndbox()

    def parse_xml(self):
        tree = ET.parse(self.xml_file)
        return tree.getroot()

    def _get_names(self):
        names = []
        for obj in self._objects:
            name = obj.find("name")
            names.append(name.text)

        return np.array(names)

    def _get_bndbox(self):
        boxes = []
        for obj in self._objects:
            coordinates = []
            bndbox = obj.find("bndbox")
            coordinates.append(np.int32(bndbox.find("xmin").text))
            coordinates.append(np.int32(np.float32(bndbox.find("ymin").text)))
            coordinates.append(np.int32(bndbox.find("xmax").text))
            coordinates.append(np.int32(bndbox.find("ymax").text))
            boxes.append(coordinates)

        return np.array(boxes)


def xml_files_to_df(xml_files):
    names = []
    boxes = []
    image_id = []
    xml_path = []
    img_path = []
    for f in xml_files:
        xml = XMLParser(f)
        names.extend(xml.names)
        boxes.extend(xml.boxes)
        image_id.extend([xml.image_id] * len(xml.names))
        xml_path.extend([xml.xml_file] * len(xml.names))
        img_path.extend([xml.img_path] * len(xml.names))
    a = {"image_id": image_id,
         "names": names,
         "boxes": boxes,
         "xml_path": xml_path,
         "img_path": img_path}

    df = pd.DataFrame.from_dict(a, orient='index')
    df = df.transpose()

    df['xmin'] = -1
    df['ymin'] = -1
    df['xmax'] = -1
    df['ymax'] = -1

    df[['xmin', 'ymin', 'xmax', 'ymax']] = np.stack([df['boxes'][i] for i in range(len(df['boxes']))])

    df.drop(columns=['boxes'], inplace=True)
    df['xmin'] = df['xmin'].astype('float32')
    df['ymin'] = df['ymin'].astype('float32')
    df['xmax'] = df['xmax'].astype('float32')
    df['ymax'] = df['ymax'].astype('float32')

    df['id'] = df['image_id'].map(lambda x: x.split(".jpg")[0])

    return df


# Working with dataframe
def build_df(xml_files):
    df = xml_files_to_df(xml_files)

    classes = sorted(df['names'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    df['label'] = df['names'].map(cls2id)

    df = df[['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']]

    return df, classes


def split_df(df, n_folds=5, training_fold=0):
    mapping = df.groupby("id")['img_path'].agg(len).to_dict()
    df['stratify'] = df['id'].map(mapping)

    kfold = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=42)

    for i, (_, val_idx) in enumerate(kfold.split(df, y=df['stratify'], groups=df['id'])):
        df.loc[val_idx, 'fold'] = i

    train_df = df[df['fold'] != training_fold].reset_index(drop=True)
    valid_df = df[df['fold'] == training_fold].reset_index(drop=True)

    return train_df, valid_df


# Working with dataset
def get_transform_train(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_transform_valid(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class VOCDataset(Dataset):
    def __init__(self, df, transforms=None, tokenizer=None):
        self.ids = df['id'].unique()
        self.df = df
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = self.df[self.df['id'] == self.ids[idx]]
        img_path = sample['img_path'].values[0]

        img = cv2.imread(img_path)[..., ::-1]
        labels = sample['label'].values
        bboxes = sample[['xmin', 'ymin', 'xmax', 'ymax']].values

        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        img = torch.FloatTensor(img).permute(2, 0, 1)

        if self.tokenizer is not None:
            seqs = self.tokenizer(labels, bboxes)
            seqs = torch.LongTensor(seqs)
            return img, seqs

        return img, labels, bboxes

    def __len__(self):
        return len(self.ids)


# Collate function
def collate_fn(batch, max_len, pad_idx):
    image_batch, seq_batch = [], []
    for image, seq in batch:
        image_batch.append(image)
        seq_batch.append(seq)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch


def get_loaders(train_df, valid_df, tokenizer, img_size, batch_size, max_len, pad_idx, num_workers=2):
    train_ds = VOCDataset(train_df, transforms=get_transform_train(
        img_size), tokenizer=tokenizer)

    trainloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_ds = VOCDataset(valid_df, transforms=get_transform_valid(
        img_size), tokenizer=tokenizer)

    validloader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=2,
        pin_memory=True,
    )

    return trainloader, validloader
