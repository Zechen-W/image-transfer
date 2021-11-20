- config.json:配置文件
- train_wzc.py:训练模型
- test_wzc.py:测试模型。输出原图像和经过*编码、信道传输、解码*后接受到的图像
- 配置文件说明：
  - cuda: 可用的GPU。如果用CPU，将此变量设为空字符串。当有多个GPU时，进行多卡训练。如果想单卡训练，
  输入指定的GPU。
  - patience: 在验证集上连续 _patience_ 个epoch性能没有提升，提前结束训练。
  - dataset_path: CIFAR-10数据集路径。注意，应为存放*data_batch_1*等文件的路径。
  - channel_param: 信噪比SNR。改变此参数不必要重新训练模型。
  - encoder_complex: 图像编码后序列长度。此参数取512时压缩率r=1/6.
  - trainable_part: 训练模型的哪个部分。取值为1或2。第一部分为编解码部分，第二部分为分类器。由于是不同的任务，
  所以分开训练。在测试时应指定为1以便计算PSNR。
  - pretrained_model: 预训练的模型，应为模型的参数文件的路径。当重新训练时此项取空字符串。
- to do:
  - [x] multi gpu or cpu adaption
  - [x] recover image normalization
  - [x] partition dataset: train, valid, test
  - [ ] save last _n_ models
  - [x] rebuild classification network: better use SOTA model
  - [ ] add log


- 注意： 

  在测试时，由于PSNR相对于MSE是下凸函数，在PSNR上平均会比在MSE上平均得到的测量指标高。这里为了数据好看，在PSNR上取平均。