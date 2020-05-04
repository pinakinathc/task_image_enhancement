import tensorflow as tf 
import time
import os
import cv2
import numpy as np
from restore_models import restore_from_dir
from unet import unet
from east_loss import east_loss, east_vis
import icdar
import matplotlib.pyplot as plt
from data import *
tf.app.flags.DEFINE_integer('max_steps', 30000, '')
tf.app.flags.DEFINE_integer('summary_save_steps', 10, '')
tf.app.flags.DEFINE_integer('model_sample_steps', 50, '')
tf.app.flags.DEFINE_integer('model_save_steps', 100, '')
tf.app.flags.DEFINE_string('checkpoint_unet', 'models/unet_model/', '')
tf.app.flags.DEFINE_string('checkpoint_east', 'models/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('checkpoint_joint_model', 'models/joint_model/', '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 7, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_integer('input_size', 256, '')
tf.app.flags.DEFINE_float('base_lr', 0.001, '')
FLAGS = tf.app.flags.FLAGS

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
		# output_images = input_images
		output_images = tf.multiply(input_enhancement_mask, output_images)
		trainable_variables = tf.trainable_variables()
		# print (trainable_variables)

		global_step = tf.Variable(0, trainable=False)

		loss, f_score, f_geometry = east_loss(sess, output_images,
			input_score_maps, input_geo_maps, input_training_masks)
		tf.summary.scalar('EAST loss', loss)
		summary_op = tf.summary.merge_all()

		# lr = tf.train.exponential_decay(FLAGS.base_lr, global_step, 100, 0.96)
		lr = 1e-4
		optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
		train_op = optimizer.minimize(loss, var_list=trainable_variables,
			global_step=global_step)

		sess.run(tf.global_variables_initializer())
		restore_from_dir(sess, FLAGS.checkpoint_unet)
		restore_from_dir(sess, FLAGS.checkpoint_east)

		data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
			input_size=FLAGS.input_size, batch_size=FLAGS.batch_size_per_gpu)

		saver = tf.train.Saver()
		summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_joint_model,
			tf.get_default_graph())
		# data_gen_args = dict(rotation_range=0.2,
		# 					width_shift_range=0.05,
		# 					height_shift_range=0.05,
		# 					shear_range=0.05,
		# 					zoom_range=0.05,
		# 					horizontal_flip=True,
		# 					fill_mode='nearest')
		# myGene = trainGenerator(10,'../training_data','night_images','day_images',data_gen_args,save_to_dir = None)


		start_step = sess.run(global_step)
		for step in range(start_step, FLAGS.max_steps):
			# img, mask = next(myGene)
			# img = img[0]
			# img[50:, 50:, :] = 0
			data = next(data_generator)

			sess.run(train_op, feed_dict={input_images:data[0],
				input_enhancement_mask:data[1], input_score_maps:data[2],
				input_geo_maps:data[3], input_training_masks:data[4]})


			if step%FLAGS.summary_save_steps==0:
				summary_str, loss_scalar = sess.run([summary_op, loss], feed_dict={input_images:data[0],
				input_enhancement_mask:data[1], input_score_maps:data[2],
				input_geo_maps:data[3], input_training_masks:data[4]})
				summary_writer.add_summary(summary_str, global_step=step)

				print ("Step: {}, Loss: {}".format(step, loss_scalar))

			if step%FLAGS.model_sample_steps==0:
				img_icdar = data[0][0]
				im_enhancement_mask = data[1][0]
				output_img = sess.run(output_images, feed_dict={input_images:[img_icdar],
					input_enhancement_mask:[im_enhancement_mask]})
				print ("sample shape: ", output_img[0].shape, output_img[0].min(), output_img[0].max())
				cv2.imwrite(os.path.join("samples", str(step)+'.png'), (output_img[0]*255.).astype(np.int32)[:, :, ::-1])
				# fig, ax = plt.subplots(1, 3)
				# ax[0].imshow((img*255).astype(np.int32))
				# ax[1].imshow((img_icdar).astype(np.int32))
				# ax[2].imshow((output_img[0]*255.).astype(np.int32))
				# plt.show()

			if step%FLAGS.model_save_steps==0:
				saver.save(sess, FLAGS.checkpoint_joint_model+'model.ckpt',
					global_step=global_step)

tf.app.run()