# Deeplabv3+ for semantic segmentation
Here we use Deeplabv3+ for body segmentation task.
References: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. [[link]](https://arxiv.org/abs/1802.02611)

## VanillaResnet 50
We use Resnet50 as the backbone for Deeplabv3+, but in a naive way. We use the first three blocks of Resnet50 and use no atrous convolution to go deeper (like in the paper). Achieve IoU = 0.9001 on the public test.
Training details: training - validation set split. Using Cross Entropy loss for each pixel. Training with SGD optimizer lr = 0.001, momentum=0.9, lr decrease 10 times after 2 epochs, training for 7 epochs.
The weight of the training model is file weight.h5
