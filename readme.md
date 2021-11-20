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
  - test_fig_output_dir: _test.py_ 输出图像的目录。
  - n_test_output_fig: _test.py_ 输出图像的数目。
# Quick Start
  ###1. Clone this repo.  
  With git, run `git clone https://github.com/Zechen-W/image-transfer.git`;  
  or alternatively, download and unzip the code via GitHub.
  ###2. Configure the environment.  
  First customize the information in *requirements.yaml*, such as _name_ and _prefix_.
  Then in commend line, run this command:  
  `conda env create -f requirements.yaml`  
  Suppose you didn't change the _name_ field, then you should activate your environment via this command:  
  `source activate wzc`
  ###3. Train the model.  
  Custom the _config.json_ file according to the instruction above. First set _trainable_part_ to
  _1_, set _pretrained_model_ to ""(a null string) and run `python train_wzc.py` to train the first part of the model, which is also known as
  the encoder and decoder part. Next you set _trainable_part_ to _2_, set _pretrained_model_ to the path referring to 
  the model state dict file which should be named _best.th_ in the _output_dir_ directory,
  and run `python train_wzc.py`. The 
  difference is that this time you will train the classifier part. Note that during the above 2
  steps, if you want to get some satisfying or even SOTA evaluation results on test set, you may 
  have to apply some training tricks such as alter the _learning_rate_ or _channel_param_ sometimes
  in the process of training.
  ###4. Evaluating the model on the test set.
  Set _pretrained_model_ to the model file directory and make other necessary changes to _config.json_.
  Run `python test_wzc.py`. The processed and the raw images will be generated, and the PSNR metric will be
  printed. To get PSNR under different SNR conditions, change _channel_param_ in _config.json_ and run
  `python test_wzc.py` repeatedly. 
  
- to do:
  - [x] multi gpu or cpu adaption
  - [x] recover image normalization
  - [x] partition dataset: train, valid, test
  - [ ] save last _n_ models
  - [x] rebuild classification network: better use SOTA model
  - [ ] add log


- 注意： 

  在测试时，由于PSNR相对于MSE是下凸函数，在PSNR上平均会比在MSE上平均得到的测量指标高。这里为了数据好看，在PSNR上取平均。