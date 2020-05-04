import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import model
from icdar import restore_rectangle
import lanms
from keras import backend as K
import time
import cv2
import os

def east_loss(sess, images, score_maps, geo_maps, training_masks):
	f_score, f_geometry = model.model(images, is_training=False)
	model_loss = model.loss(score_maps, f_score,
		geo_maps, f_geometry, training_masks)
	total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	return total_loss, f_score, f_geometry

def sort_poly(p):
	min_axis = np.argmin(np.sum(p, axis=1))
	p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
	if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
		return p
	else:
		return p[[0, 3, 2, 1]]

def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
	'''
	restore text boxes from score map and geo map
	:param score_map:
	:param geo_map:
	:param timer:
	:param score_map_thresh: threshhold for score map
	:param box_thresh: threshhold for boxes
	:param nms_thres: threshold for nms
	:return:
	'''
	if len(score_map.shape) == 4:
		score_map = score_map[0, :, :, 0]
		geo_map = geo_map[0, :, :, ]
	# filter the score map
	xy_text = np.argwhere(score_map > score_map_thresh)
	# sort the text boxes via the y axis
	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	# restore
	start = time.time()
	text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
	# print('{} text boxes before nms'.format(text_box_restored.shape[0]))
	boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = text_box_restored.reshape((-1, 8))
	boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
	timer['restore'] = time.time() - start
	# nms part
	start = time.time()
	# boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
	timer['nms'] = time.time() - start

	if boxes.shape[0] == 0:
		return None, timer

	# here we filter some low score boxes by the average score map, this is different from the orginal paper
	for i, box in enumerate(boxes):
		mask = np.zeros_like(score_map, dtype=np.uint8)
		cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
		boxes[i, 8] = cv2.mean(score_map, mask)[0]
	boxes = boxes[boxes[:, 8] > box_thresh]

	return boxes, timer

if __name__ == "__main__":
	import icdar
	import os

	data_generator = icdar.get_batch(num_workers=1,
			input_size=256, batch_size=1)

	input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
		name='input_images')
	input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1],
		name='input_score_maps')
	input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5],
		name='input_geo_maps')
	input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1],
		name='input_training_masks')
	input_enhancement_mask = tf.placeholder(tf.float32, shape=[None, None, None, 3],
		name='input_images')

	# f_score, f_geometry = model.model(input_images, is_training=False)	
	with tf.Session() as sess:
		# restore_from_dir(sess, FLAGS.checkpoint_east)
		sess.run(tf.global_variables_initializer())
		loss, f_score, f_geometry = east_loss(sess, input_images, input_score_maps,
			input_geo_maps, input_training_masks)	
		# f_score, f_geometry = model.model(input_images, is_training=False)			
		# restore_from_dir(sess, FLAGS.checkpoint_east)
		for i in range(10):
			data = next(data_generator)
			start_time = time.time()
			im = data[0][0].astype(np.int32)[:, :, ::-1]
			im_resized = im

			timer = {'net': 0, 'restore': 0, 'nms': 0}
			start = time.time()
			score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
			timer['net'] = time.time() - start

			boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
			
			if boxes is not None:
				boxes = boxes[:, :8].reshape((-1, 4, 2))
				# boxes[:, :, 0] /= ratio_w
				# boxes[:, :, 1] /= ratio_h

			duration = time.time() - start_time
			print('[timing] {}'.format(duration))

			if boxes is not None:
				for box in boxes:
					# to avoid submitting errors
					box = sort_poly(box.astype(np.int32))
					if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
						continue
					cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)			
				print (im.shape, score.shape)
				plt.imshow(im[:, :, ::-1])
				plt.show()
