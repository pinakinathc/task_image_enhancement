import tensorflow as tf 
import time
import os
import cv2
import numpy as np
from restore_models import restore_from_dir
from unet import unet
from east_loss import east_loss
import eval_data_loader as data_loader
import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('validation_data_path', 'samples/', '')
tf.app.flags.DEFINE_string('checkpoint_joint_model', 'models/joint_model_real/', '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 1, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
FLAGS = tf.app.flags.FLAGS

def change_img(img):
	M, N, _ = img.shape
	M_, N_, = M-1, N-1
	a = np.sum(np.sum(img, axis=1), axis=1)
	while(True):
		if a[M_] == 0:
			M_ -= 1
		else:
			M_ += 1
			break
	a = np.sum(np.sum(img, axis=0), axis=1)
	while(True):
		if a[N_] == 0:
			N_ -= 1
		else:
			N_ += 1
			break
	print (M, N, M_, N_)
	return img[:M_, :N_, :]

def main(argv):
	with tf.Session() as sess:
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
			name='input_images')
		input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1],
			name='input_score_maps')
		input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5],
			name='input_geo_maps')
		input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1],
			name='input_training_masks')
		input_enhancement_mask = tf.placeholder(tf.float32, shape=[None, None, None, 3],
			name='input_enhancement_mask')

		output_images = tf.multiply(unet(tf.divide(input_images, 255.)), 255.)
		output_images = tf.multiply(input_enhancement_mask, output_images)
		global_step = tf.Variable(0, trainable=False)

		loss, f_score, f_geometry = east_loss(sess, output_images,
			input_score_maps, input_geo_maps, input_training_masks)

		restore_from_dir(sess, FLAGS.checkpoint_joint_model)
		
		data_generator = data_loader.get_batch(num_workers=FLAGS.num_readers,
			input_size=FLAGS.input_size, batch_size=FLAGS.batch_size_per_gpu,
			image_path=FLAGS.validation_data_path, validation=True)

		for i in range(400):
			data = next(data_generator)
			img_icdar = data[0][0]
			im_enhancement_mask = data[1][0]
			im, score, geometry = sess.run([output_images, f_score, f_geometry],
				feed_dict={input_images:[img_icdar],
					input_enhancement_mask:[im_enhancement_mask]})
			im = (im[0]).astype(np.int32)[:, :, ::-1]
			file_name = os.path.split(data[6][0])[-1]
			# cv2.imwrite(os.path.join('sample_outputs/enhancement_output/unet', file_name),
			# 	change_img(im))
			plt.imshow(im[:, :, ::-1])
			plt.show()

tf.app.run()
