- config.json:配置文件
- train_wzc.py:训练模型
- test_wzc.py:测试模型。输出原图像和经过*编码、信道传输、解码*后接受到的图像
- 配置文件说明：
  - cuda: 可用的GPU。如果用CPU，将此变量设为空字符串。当有多个GPU时，进行多卡训练。如果想单卡训练，
  输入指定的GPU。
  - patience: 在验证集上连续 _patience_ 个epoch性能没有提升，提前结束训练。
  - channel_param: 信噪比SNR。改变此参数不必要重新训练模型。
  - encoder_complex: 图像编码后序列长度。此参数取512时压缩率r=1/6.
  - trainable_part: 训练模型的哪个部分。取值为1或2。第一部分为编解码部分，第二部分为分类器。由于是不同的任务，
  所以分开训练。
- to do:
  - [x] multi gpu or cpu adaption
  - [x] recover image normalization
  - [ ] partition dataset: train, valid, test
  - [ ] save last _n_ models
  - [x] rebuild classification network: better use SOTA model
- thanks to:
  - https://github.com/weiaicunzai/pytorch-cifar100 , where ResNet used in this project comes from.

