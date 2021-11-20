from PIL import Image
import pickle
import os
from torch.utils.data import Dataset
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


class WzcDataset(Dataset):
    def __init__(self, root, dataset_type, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if dataset_type not in ['train', 'valid', 'test']:
            raise RuntimeError('argument dataset_type must be one of ("train", "valid", test")')
        data = []
        labels = []
        filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

        for filename in filenames:
            batch_data = unpickle(os.path.join(root, filename))
            data.append(batch_data[b'data'])
            labels.extend(batch_data[b'labels'])
        data = np.vstack(data).reshape((-1, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))

        np.random.seed(666)
        np.random.shuffle(data)
        np.random.shuffle(data)
        np.random.seed(666)
        np.random.shuffle(labels)
        np.random.shuffle(labels)

        if dataset_type == 'train':
            self.data = data[:48000]
            self.labels = labels[:48000]
        elif dataset_type == 'valid':
            self.data = data[48000:54000]
            self.labels = labels[48000:54000]
        else:
            self.data = data[54000:]
            self.labels = labels[54000:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, label = self.data[item], self.labels[item]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label
