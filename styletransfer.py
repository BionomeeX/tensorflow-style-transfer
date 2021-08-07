import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

parser = argparse.ArgumentParser(
    usage="%(Style Transfer)s [OPTION] -I image 0 -J image 1",
    description="Apply style transfer",
)
parser.add_argument(
    "-I", help="Input image 0", type=str
)
parser.add_argument(
    "-J", help="Input image 1", type=str
)
args = parser.parse_args()

img0 = load_img(args.I)
img1 = load_img(args.J)


print(img0.shape)
print(img1.shape)


image0 = hub_model(tf.constant(img0), tf.constant(img1))[0]
image1 = hub_model(tf.constant(img1), tf.constant(img0))[0]

outfileI = ".".join(os.path.basename(args.I).split(".")[:-1]) + ".jpg"
outfileJ = ".".join(os.path.basename(args.J).split(".")[:-1]) + ".jpg"

tf.io.write_file(outfileI, tf.image.encode_jpeg(tf.image.convert_image_dtype(image0[0,], tf.uint8)))
tf.io.write_file(outfileJ, tf.image.encode_jpeg(tf.image.convert_image_dtype(image1[0,], tf.uint8)))
