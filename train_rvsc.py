#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip

from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay

seed = 1234
np.random.seed(seed)

RVSC_ROOT_PATH = 'RVSC_data'

TRAIN_PATH = os.path.join(RVSC_ROOT_PATH, 'TrainingSet')

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'P(\d{02})-(\d{04})-.*', ctr_path)
        self.patient_no = match.group(1)
        self.img_no = match.group(2)


def read_contour(contour, data_path, return_mask=True):
    img_path = [dirpath for dirpath, dirnames, files in os.walk(data_path)
                if contour.patient_no+'dicom' in dirpath][0]
    filename = 'P{:s}-{:s}.dcm'.format(contour.patient_no, contour.img_no)
    full_path = os.path.join(img_path, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    if img.ndim < 3:
        img = img[..., np.newaxis]
    if not return_mask:
        return img, None
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if mask.ndim < 3:
        mask = mask[..., np.newaxis]

    return img, mask


def map_all_contours(data_path, contour_type, shuffle=True):
    list_files = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in files if 'list' in f]
    contours = []
    for f in list_files:
        for line in open(f).readlines():
            line = line.strip().replace('\\','/')
            full_path = os.path.join(data_path, line)
            if contour_type+'contour' in full_path:
                contours.append(full_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path, return_mask=True)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks


if __name__== '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 200

    print('Mapping ground truth '+contour_type+' contours to images in train...')
    train_ctrs = map_all_contours(TRAIN_PATH, contour_type, shuffle=True)
    print('Done mapping training set')
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    print('\nBuilding train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_PATH,
                                                crop_size=crop_size)
    print('\nBuilding dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            TRAIN_PATH,
                                            crop_size=crop_size)

    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    model = fcn_model(input_shape, num_classes, weights=None)

    kwargs = dict(
        rotation_range=0,
        zoom_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
    )
    image_datagen = ImageDataGenerator(**kwargs)
    mask_datagen = ImageDataGenerator(**kwargs)

    epochs = 20
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = izip(image_generator, mask_generator)

    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        print('\nLearning rate: {:6f}\n'.format(lrate))
        train_result = []
        for iteration in range(len(img_train)/mini_batch_size):
            img, mask = next(train_generator)
            res = model.train_on_batch(img, mask)
            curr_iter += 1
            lrate = lr_poly_decay(model, base_lr, curr_iter,
                                  max_iter, power=0.5)
            train_result.append(res)
        train_result = np.asarray(train_result)
        train_result = np.mean(train_result, axis=0).round(decimals=10)
        print('Train result {:s}:\n{:s}'.format(model.metrics_names, train_result))
        print('\nEvaluating dev set ...')
        result = model.evaluate(img_dev, mask_dev, batch_size=32)
        result = np.round(result, decimals=10)
        print('\nDev set result {:s}:\n{:s}'.format(model.metrics_names, result))
        save_file = '_'.join(['rvsc', contour_type,
                              'epoch', str(e+1)]) + '.h5'
        if not os.path.exists('model_logs'):
            os.makedirs('model_logs')
        save_path = os.path.join('model_logs', save_file)
        print('\nSaving model weights to {:s}'.format(save_path))
        model.save_weights(save_path)


