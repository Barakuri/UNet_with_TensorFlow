import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Convolution2DTranspose, MaxPool2D
from tensorflow.keras import Model
import numpy as np

img_h, img_w = 572, 572
img_c = 1
image = tf.keras.Input((img_h, img_w, img_c))


def double_conv(filters, kernel_size):
    conv = tf.keras.Sequential()
    conv.add(Conv2D(filters, kernel_size, activation='relu'))
    conv.add(Conv2D(filters, kernel_size, activation='relu'))
    return conv


def double_conv_decoder(filters, kernel_size):
    conv = tf.keras.Sequential()
    conv.add(Conv2D(filters, kernel_size, activation='relu'))
    conv.add(Conv2D(filters, kernel_size, activation='relu'))
    return conv


def up_conv(filters, kernel_size, stride):
    return Convolution2DTranspose(filters, kernel_size, stride)


def poolinglayer(kernel_size, stride):
    return MaxPool2D((kernel_size, kernel_size), stride)


def crop_img(encoder, decoder):
    decoder_size = decoder.shape[2]
    encoder_size = encoder.shape[2]
    delta = encoder_size - decoder_size
    delta = delta // 2
    return encoder[:, delta:encoder_size - delta, delta:encoder_size - delta, :]


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_conv_1 = double_conv(64, 3)
        self.max_pool = poolinglayer(2, 2)
        self.down_conv_2 = double_conv(128, 3)
        self.down_conv_3 = double_conv(256, 3)
        self.down_conv_4 = double_conv(512, 3)
        self.down_conv_5 = double_conv(1024, 3)
        self.up_conv_1 = up_conv(512, 2, 2)
        self.up_conv_2 = up_conv(256, 2, 2)
        self.up_conv_3 = up_conv(128, 2, 2)
        self.up_conv_4 = up_conv(64, 2, 2)
        self.decoder_conv1 = double_conv_decoder(512, 3)
        self.decoder_conv2 = double_conv_decoder(256, 3)
        self.decoder_conv3 = double_conv_decoder(128, 3)
        self.decoder_conv4 = double_conv_decoder(64, 3)

    def call(self, image):
        x1 = self.down_conv_1(image)
        x2 = self.max_pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool(x7)
        x9 = self.down_conv_5(x8)

        x10 = self.up_conv_1(x9)
        y1 = crop_img(x7, x10)
        x11 = self.decoder_conv1(tf.concat([x10, y1], axis=3))

        x12 = self.up_conv_2(x11)
        y2 = crop_img(x5, x12)
        x13 = self.decoder_conv2(tf.concat([x12, y2], axis=3))

        x14 = self.up_conv_3(x13)
        y3 = crop_img(x3, x14)
        x15 = self.decoder_conv3(tf.concat([x14, y3], axis=3))

        x16 = self.up_conv_4(x15)
        y4 = crop_img(x1, x16)
        x17 = self.decoder_conv4(tf.concat([x16, y4], axis=3))

        x18 = Conv2D(2, 1)(x17)
        return x18


if __name__ == "__main__":
    image = np.random.standard_normal(size=(1, 572, 572, 1))
    model = UNet()
    print(model(image))
