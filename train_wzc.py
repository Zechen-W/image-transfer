import json
import os

with open('./config.json', encoding='utf8') as r:
    args = json.load(r)
os.environ['CUDA_VISIBLE_DEVICE'] = args['cuda']

import torch
from models import WZCModel
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
import re
from datasets import WzcDataset


def main():
    with open(os.path.join(args['output_dir'], 'config.json'), 'w') as w:
        json.dump(args, w, indent=4)

    patience = args['patience']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    train_data = WzcDataset(args['dataset_path'], 'train', transform=transform)
    valid_data = WzcDataset(args['dataset_path'], 'valid', transform=transform)

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=True)
    model = WZCModel(K=args['encoder_complex'],
                     channel_type=args['channel_type'],
                     channel_param=args['channel_param'],
                     trainable_part=args['trainable_part'])
    if args['pretrained_model']:
        origin_state_dict = torch.load(args['pretrained_model'])
        state_dict = {}
        for key, value in origin_state_dict.items():
            key = re.sub(r'^module\.', '', key)
            state_dict[key] = value
        model.load_state_dict(state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(args['cuda']) > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    best_eval_metric = 0
    if args['trainable_part'] == 1:
        best_eval_metric = 1e9
    patience_timer = 0

    for epoch in range(args['num_epochs']):

        # training
        model.train()
        n_batch = len(train_loader)
        batch_i = 0
        sumLoss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            mid_out, final_out, loss = model(images, labels)
            sumLoss += loss.sum()
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            batch_i += 1
            if batch_i % 50 == 0 or batch_i == n_batch:
                print(f'Train Epoch : {epoch:02} [{batch_i:4}/{n_batch:4} ({100. * batch_i / n_batch:3.0f}%)] '
                      f'loss:{loss.sum() / args["batch_size"]:.6f}')

        print(f"average loss on train set: {sumLoss / len(train_data)}")

        # evaluating
        model.eval()
        eval_acc = 0
        eval_loss = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                clf_outputs, losses = model(images, labels)[1:]
                # calculate acc
                y_hat = clf_outputs.argmax(dim=-1)
                eval_correct = (y_hat == labels.cuda()).sum()
                eval_acc += eval_correct.item()

                # calculate loss
                eval_loss += losses.sum()

        eval_acc /= len(valid_loader)
        eval_loss /= len(valid_loader)
        print(f'eval_acc: {eval_acc}, eval_loss: {eval_loss}')

        # save model and early stop
        if args['trainable_part'] == 1:
            if eval_loss < best_eval_metric:
                print(f'best performance on valid set. saving model to {args["output_dir"] + "best.th"}.')
                best_eval_metric = eval_loss
                patience_timer = 0
                torch.save(model.state_dict(), os.path.join(args['output_dir'], 'best.th'))
            else:
                patience_timer += 1
        else:
            if eval_acc > best_eval_metric:
                print(f'best performance on valid set. saving model to {args["output_dir"] + "best.th"}.')
                best_eval_metric = eval_acc
                patience_timer = 0
                torch.save(model.state_dict(), os.path.join(args['output_dir'], 'best.th'))
            else:
                patience_timer += 1

        if patience_timer == patience:
            print('run out of patience, early stopping')
            break


if __name__ == '__main__':
    main()
