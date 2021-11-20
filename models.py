from torchvision.models import resnet50
import torch
import torch.nn as nn
import numpy as np
from typing import Union


class Encoder(nn.Module):
    def __init__(self, K):
        super(Encoder, self).__init__()
        filters = (16, 32, 32, 32)
        # thr num of complex after mapping

        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        # (3,32,32) -> (16,16,16), with implicit padding
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=(5, 5), stride=2, padding=(5 - 2) // 2 + 1),
            nn.ReLU()
        )

        # (16,16,16) -> (32,8,8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=(5, 5), stride=2, padding=(5 - 2) // 2 + 1),
            nn.ReLU()
        )

        # (32,8,8) -> (32,8,8)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            nn.ReLU()
        )

        # (32,8,8) -> (32,8,8)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            nn.ReLU()
        )

        # (32,8,8) -> (2K), K complex
        self.linear = nn.Linear(32 * 8 * 8, 2 * K)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.reshape(-1, 32 * 8 * 8)
        out = self.linear(x)
        return out


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, channel_type, channel_param):
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.channel_param = channel_param

    def gaussian_noise_layer(self, input_layer, std):
        noise = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise.to(input_layer.get_device())
        return input_layer + noise

    def normalize(self, x):
        # B C H W
        pwr = torch.mean(x ** 2, (-2, -1), True)  # * 2
        return x / torch.sqrt(pwr)

    def forward(self, channel_in):

        # power normalization
        channel_in = self.normalize(channel_in)

        if self.channel_type == 0 or self.channel_type == 'none':
            return channel_in

        elif self.channel_type == 1 or self.channel_type == 'awgn':
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.channel_param / 10)))
            output = self.gaussian_noise_layer(channel_in, std=sigma)
            return output


class Decoder(nn.Module):
    def __init__(self, K):
        super(Decoder, self).__init__()

        filters = [32, 32, 32, 16]

        self.linear = nn.Linear(2 * K, 32 * 8 * 8)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[1], kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            nn.ReLU()
        )

        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[2], kernel_size=(5, 5), stride=1, padding=(5 - 1) // 2),
            nn.ReLU()
        )

        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[3], kernel_size=(4, 4), stride=2, padding=(5 - 2) // 2),
            nn.ReLU()
        )

        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], 3, kernel_size=(4, 4), stride=2, padding=(5 - 2) // 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x.reshape(-1, 32, 8, 8)
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        x = self.upconv_block4(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 32 * 32 * 3)  # 全连接层
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(32 * 32 * 3, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        max_rgb = torch.max(x.reshape(batch_size, 3, 32 * 32), -1)[0]
        x = x.T.div(max_rgb.T).T
        return x


class WZCModel(torch.nn.Module):
    def __init__(self, K: int,
                 channel_type: str,
                 channel_param: Union[int, float],
                 trainable_part: int):
        super(WZCModel, self).__init__()

        self.encoder = Encoder(K)
        self.channel = Channel(channel_type, channel_param)
        self.decoder = Decoder(K)
        self.normalizer = Normalizer()
        self.classifier = torch.nn.Sequential(resnet50(pretrained=trainable_part == 1),
                                              torch.nn.Linear(1000, 10))
        self.distortion_loss = torch.nn.MSELoss()
        self.classify_loss = torch.nn.CrossEntropyLoss()
        self.trainable_part = trainable_part

    def forward(self, input_image, label):
        assert self.trainable_part in [1, 2], "trainable part must be 1 or 2"
        if self.trainable_part == 1:
            feature = self.encoder(input_image)
            noisy_feature = self.channel(feature)
            recover_image = self.normalizer(self.decoder(noisy_feature))
            classify_out = torch.softmax(self.classifier(recover_image), 0)
            distortion_loss = self.distortion_loss(input_image, recover_image)

            return recover_image, classify_out, distortion_loss
        else:
            with torch.no_grad():
                feature = self.encoder(input_image)
                noisy_feature = self.channel(feature)
                recover_image = self.normalizer(self.decoder(noisy_feature))
            classify_out = self.classifier(recover_image)
            classify_loss = self.classify_loss(classify_out, label)
            return recover_image, classify_out, classify_loss
