# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

import tensorflow as tf

from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('valid_data_path', '../training_data/night_images',
                           'training dataset to use')

FLAGS = tf.app.flags.FLAGS


def get_images(image_path):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(image_path, '*.{}'.format(ext))))
    return files


def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False, image_path=FLAGS.valid_data_path, validation=False):
    image_list = np.array(get_images(image_path))
    if validation:
        image_list.sort()
    print('{} training images in {}'.format(
        image_list.shape[0], image_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        if not validation:
            np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        im_padded_enhancs = []
        all_text_polys = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                # print im_fn
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded_enhanc = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im_padded_enhanc[:new_h, :new_w, :] = (np.ones_like(im)*1).copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                im_padded_enhanc = cv2.resize(im_padded_enhanc, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                new_h, new_w, _ = im.shape

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                im_padded_enhancs.append(im_padded_enhanc[:, :, ::-1].astype(np.float32))

                if len(images) == batch_size:
                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield images, im_padded_enhancs, score_maps, geo_maps, training_masks, all_text_polys, image_fns
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
                    im_padded_enhancs = []
                    all_text_polys = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=False)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
