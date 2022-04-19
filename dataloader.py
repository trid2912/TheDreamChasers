import tensorflow as tf
import json

with open("train.json", "r") as f:
    train_list, val_list = json.load(f)

img_dir = ""
mask_dir = ""

def data_generator(img_list, bs):
    dataset = tf.data.Dataset.from_tensor_slices(img_list)
    dataset = dataset.shuffle(2010, reshuffle_each_iteration=True)
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(bs, drop_remainder=False)
    return dataset

def load_data(img_name):
    img = read_image(img_name)
    mask = read_mask(img_name)
    return img, mask

def read_image(name):
    img = tf.io.read_file(img_dir + name)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1
    return img

def read_mask(name):
    img = tf.io.read_file(mask_dir + name)
    img = tf.image.decode_png(img, channels=1)
    img = img[:, :, 0] // 255
    img = tf.one_hot(img, 2)
    return img

trainloader = data_generator(train_list, 6)
valloader = data_generator(val_list, 6)
