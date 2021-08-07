import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


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

img0 = tf.io.read_file(args.I)
img0 = tf.image.decode_image(img0, channels=3)
img0 = tf.image.convert_image_dtype(img0, tf.float32)
img1 = tf.io.read_file(args.J)
img1 = tf.image.decode_image(img1, channels=3)
img1 = tf.image.convert_image_dtype(img1, tf.float32)

image0 = hub_model(tf.constant(img0), tf.constant(img1))[0]
image1 = hub_model(tf.constant(img1), tf.constant(img0))[0]

outfile = ".".join(os.path.basename(args.I).split(".")[:-1])

tf.io.write_file(outfile + "_0.jpg", tf.image.encode_jpeg(tf.image.convert_image_dtype(image0[0,], tf.uint8)))
tf.io.write_file(outfile + "_1.jpg", tf.image.encode_jpeg(tf.image.convert_image_dtype(image1[0,], tf.uint8)))
