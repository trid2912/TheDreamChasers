import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, AveragePooling2D, UpSampling2D, Concatenate, Layer, ReLU


class ConvolutionBlock(Layer):

    def __init__(self, num_filters, kernel_size, dilation_rate, bias):
        super(ConvolutionBlock, self).__init__()
        self.conv2d = Conv2D(num_filters, kernel_size, padding="SAME", dilation_rate=dilation_rate, use_bias=bias)
        self.batchnorm = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        output = self.relu(x)
        return output


class ImagePooling(Layer):

    def __init__(self, input_size):
        super(ImagePooling, self).__init__()
        self.avgpool = AveragePooling2D(pool_size=(input_size, input_size))
        self.conv = ConvolutionBlock(256, 1, dilation_rate=1, bias=True)
        self.upsample = UpSampling2D(size=(input_size, input_size), interpolation="bilinear")

    def call(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        out = self.upsample(x)
        return out


class ASSP(Layer):

    def __init__(self, input_size):
        super(ASSP, self).__init__()
        self.img_pooling = ImagePooling(input_size)
        self.conv1 = ConvolutionBlock(256, 1, dilation_rate=1, bias=True)
        self.conv6 = ConvolutionBlock(256, 3, dilation_rate=6, bias=True)
        self.conv12 = ConvolutionBlock(256, 3, dilation_rate=12, bias=True)
        self.conv18 = ConvolutionBlock(256, 3, dilation_rate=18, bias=True)
        self.concat = Concatenate(axis=-1)
        self.conv_out = ConvolutionBlock(256, 1, dilation_rate=1, bias=True)

    def call(self, x):
        out_pool = self.img_pooling(x)
        out_conv1 = self.conv1(x)
        out_conv6 = self.conv6(x)
        out_conv12 = self.conv12(x)
        out_conv18 = self.conv18(x)
        out_concat = self.concat([out_pool, out_conv1, out_conv6, out_conv12, out_conv18])
        out = self.conv_out(out_concat)
        return out


def r50_backbone(img_size):
    input = Input(shape=(img_size, img_size, 3))
    resnet50 = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=input)
    out16 = resnet50.get_layer("conv4_block6_out").output
    out8 = resnet50.get_layer("conv2_block3_out").output
    return tf.keras.Model(inputs=input, outputs=[out16, out8])


class DeeplabV3plus_r50(tf.keras.Model):

    def __init__(self, img_size, stride, num_classes):
        super(DeeplabV3plus_r50, self).__init__()
        self.backbone = r50_backbone(256)
        self.assp = ASSP(img_size // stride)
        self.upsample1 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.conv1 = ConvolutionBlock(48, 1, dilation_rate=1, bias=True)
        self.concat = Concatenate(axis=-1)
        self.conv2 = ConvolutionBlock(256, 3, dilation_rate=1, bias=False)
        self.upsample2 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.conv3 = Conv2D(num_classes, 1, activation="softmax")

    def call(self, x):
        feature16, feature4 = self.backbone(x)
        x = self.assp(feature16)
        x = self.upsample1(x)
        y = self.conv1(feature4)
        x = self.concat([y, x])
        x = self.conv2(x)
        x = self.upsample2(x)
        output = self.conv3(x)
        return output
