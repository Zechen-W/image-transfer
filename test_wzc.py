import torch
from models import WZCModel
import json
import os
import torch
from models import WZCModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb


def main():
    with open('./config.json', encoding='utf8') as r:
        args = json.load(r)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    data2 = datasets.CIFAR10(root=args['dataset_path'], train=False, download=False, transform=transform)
    # all_data = list(data1 + data2)
    # pdb.set_trace()
    # train_loader = DataLoader(all_data[:int(.8 * n_data)], batch_size=args['batch_size'], shuffle=False)
    # valid_loader = DataLoader(all_data[int(.8 * n_data):int(.9 * n_data)], batch_size=args['batch_size'], shuffle=False)
    # test_loader = DataLoader(all_data[int(.9 * n_data):], batch_size=args['batch_size'], shuffle=False)
    loader = DataLoader(data2, batch_size=args['batch_size'], shuffle=False)
    model = WZCModel(K=args['encoder_complex'],
                     channel_type=args['channel_type'],
                     channel_param=args['channel_param'],
                     trainable_part=args['trainable_part'])
    if args['pretrained_model']:
        model.load_state_dict(torch.load(args['pretrained_model']))
    model = model.to('cuda')

    i = 0
    errcnt = 0
    with torch.no_grad():
        for images, labels in loader:

            images, labels = images.to('cuda'), labels.to('cuda')
            aftimages, clfres = model(images, labels)[:2]
            for bef, aft in zip(images, aftimages):
                if i > 100:
                    break
                plt.imsave(f'./figs/bef{i}.jpg', bef.T.transpose(0, 1).cpu().numpy())
                try:
                    plt.imsave(f'./figs/aft{i}.jpg', aft.T.transpose(0, 1).cpu().numpy())
                except ValueError:
                    errcnt += 1
                i += 1


if __name__ == '__main__':
    main()
