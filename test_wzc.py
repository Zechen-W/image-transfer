import json
import os
import torch
from models import WZCModel
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pdb
import re
from datasets import WzcDataset


def main():
    with open('./config.json', encoding='utf8') as r:
        args = json.load(r)

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    test_data = WzcDataset(root=args['dataset_path'], dataset_type='test', transform=transform)
    loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    # load model
    model = WZCModel(K=args['encoder_complex'],
                     channel_type=args['channel_type'],
                     channel_param=args['channel_param'],
                     trainable_part=args['trainable_part'])
    origin_state_dict = torch.load(args['pretrained_model'])
    state_dict = {}
    for key, value in origin_state_dict.items():
        key = re.sub(r'^module\.', '', key)
        state_dict[key] = value
    model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    i = 0
    with torch.no_grad():
        psnr = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            aftimages, clfres, mse = model(images, labels)
            psnr += 10 * torch.log10(1/mse)
            for bef, aft in zip(images, aftimages):
                if i > 100:
                    break
                plt.imsave(os.path.join(args['test_args']['output_fig_dir'], f'bef{i}.jpg'),
                           bef.T.transpose(0, 1).cpu().numpy())
                plt.imsave(os.path.join(args['test_args']['output_fig_dir'], f'aft{i}.jpg'),
                           aft.T.transpose(0, 1).cpu().numpy())
                i += 1
        psnr /= len(loader)
        print(f'psnr:{psnr}')


if __name__ == '__main__':
    main()
