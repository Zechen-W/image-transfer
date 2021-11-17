import json
import os
import torch
from models import WZCModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pdb


def main():
    with open('./config.json', encoding='utf8') as r:
        args = json.load(r)

    with open(os.path.join(args['output_dir'], 'config.json'), 'w') as w:
        json.dump(args, w, indent=4)

    patience = args['patience']
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    data1 = datasets.CIFAR10(root=args['dataset_path'], train=True, download=False, transform=transform)
    data2 = datasets.CIFAR10(root=args['dataset_path'], train=False, download=False, transform=transform)
    # all_data = list(data1 + data2)
    n_data = len(data1 + data2)
    # pdb.set_trace()
    # train_loader = DataLoader(all_data[:int(.8 * n_data)], batch_size=args['batch_size'], shuffle=False)
    # valid_loader = DataLoader(all_data[int(.8 * n_data):int(.9 * n_data)], batch_size=args['batch_size'], shuffle=False)
    # test_loader = DataLoader(all_data[int(.9 * n_data):], batch_size=args['batch_size'], shuffle=False)
    train_loader = DataLoader(data1, batch_size=args['batch_size'], shuffle=False)
    valid_loader = DataLoader(data2, batch_size=args['batch_size'], shuffle=False)
    model = WZCModel(K=args['encoder_complex'],
                     channel_type=args['channel_type'],
                     channel_param=args['channel_param'],
                     trainable_part=args['trainable_part'])
    if args['pretrained_model']:
        model.load_state_dict(torch.load(args['pretrained_model']))
    gpus = args['cuda']
    if gpus:
        model = model.to('cuda')
        # if len(gpus) > 1:
        #     model = torch.nn.DataParallel(model)
    else:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    max_acc = 0
    patience_timer = 0
    for epoch in range(args['num_epochs']):
        model.train()
        n_batch = len(train_loader)
        batch_i = 0
        sumLoss = 0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            mid_out, final_out, loss = model(images, labels)
            sumLoss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_i += 1
            if batch_i % 50 == 0 or batch_i == n_batch:
                print(f'Train Epoch : {epoch:02} [{batch_i:4}/{n_batch:4} ({100. * batch_i / n_batch:3.0f}%)] '
                      f'loss:{loss.sum() / args["batch_size"]:.6f}')

        print(f"average loss on train set: {sumLoss / 50000:.6f}")
        model.eval()
        eval_acc = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                clf_output, loss = model(images, labels)[1:]
                y_hat = clf_output.argmax(dim=-1)
                eval_correct = (y_hat == labels.cuda()).sum()
                eval_acc += eval_correct.item()

        eval_acc /= len(valid_loader)
        print(eval_acc)
        if eval_acc > max_acc:
            print(f'best performance on valid set. saving model to {args["output_dir"] + "best.th"}.')
            max_acc = eval_acc
            patience_timer = 0
            torch.save(model.state_dict(), os.path.join(args['output_dir'], 'best.th'))
        else:
            patience_timer += 1

        if patience_timer == patience:
            print('run out of patience, early stopping')
            break


if __name__ == '__main__':
    main()
