#!/usr/bin/env python2.7

import dicom, re
import os, sys
import numpy as np
from skimage import measure, io
from multiprocessing import Pool
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip

from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay

seed = 1234
np.random.seed(seed)

LVSC_ROOT_PATH = 'LVSC_data'

TRAIN_PATH = os.path.join(LVSC_ROOT_PATH, 'Training')


def read_contour(contour_file):
    img_path = contour_file[:contour_file.find('.png')]+'.dcm'
    f = dicom.read_file(img_path)
    img = f.pixel_array.astype('int')
    ctr = io.imread(contour_file, as_grey=True).astype('int')
    labels = measure.label(ctr, background=1000, connectivity=2)
    if contour_type == 'i':
        mask = np.where(labels == 3, 1, 0)
    elif contour_type == 'o':
        mask = np.zeros_like(ctr)
        myo = np.where(labels == 2, 1, 0)
        endo = np.where(labels == 3, 1, 0)
        mask = mask + myo + endo
    else:
        mask = ctr
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask


def validate_contour(contour_file):
    ctr = io.imread(contour_file, as_grey=True).astype('int')
    labels, num = measure.label(ctr, background=0,
                                connectivity=2, return_num=True)
    if num >= 2:
        return None
    labels, num = measure.label(ctr, background=1000,
                                connectivity=2, return_num=True)
    if num != 3:
        return None
    small_ctr = np.where(labels == 3, 1, 0)
    if np.sum(small_ctr) < 25:
        return None
    
    return contour_file


def map_all_contours(data_path, shuffle=True):
    contour_files = [os.path.join(dirpath, f)
                     for dirpath, dirnames, files in os.walk(data_path)
                     for f in files if all(s in f for s in ['SA','.png'])]
    
    #contour_files = np.random.choice(contour_files, 5000)
    
    pool = Pool(8)
    results = pool.map(validate_contour, contour_files)
    pool.close()
    valid_ctrs = [valid for valid in results if valid is not None]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(valid_ctrs)
    print('Number of examples: {:d}'.format(len(valid_ctrs)))
    
    return valid_ctrs


def generate_all_contours(contours, chunk_size, crop_size, shuffle=False):
    num_iter = int(np.ceil(len(contours) / float(chunk_size)))
    while True: # this flag yields an infinite generator
        if shuffle:
            print('\nShuffling data at each epoch\n')
            np.random.shuffle(contours)
        for i in xrange(num_iter):
            chunk = contours[(chunk_size*i):(chunk_size*(i+1))]
            if len(chunk) == 0:
                break
            pool = Pool(8)
            results = pool.map(read_contour, chunk)
            pool.close()
            for (image, mask) in results:
                image = center_crop(image, crop_size=crop_size)
                mask = center_crop(mask, crop_size=crop_size)
                
                yield (image, mask)


def batch_data(dataset, batch_size):
    while True:
        images, masks = zip(*[next(dataset) for i in xrange(batch_size)])
        images = np.asarray(images)
        masks = np.asarray(masks)
        batch = (images, masks)

        yield batch


if __name__== '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: python %s <i/o/myo> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    crop_size = 200
    train_batch_size = 1000
    dev_batch_size = 100

    print('Mapping ground truth '+contour_type+' contours to images in train...')
    train_ctrs = map_all_contours(TRAIN_PATH, shuffle=True)
    print('Done mapping training set')
    split = 2000
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    print('\nGenerating train dataset ...')
    train_generator = generate_all_contours(train_ctrs,
                                            shuffle=False,
                                            crop_size=crop_size,
                                            chunk_size=train_batch_size)
    train_batch = batch_data(train_generator, batch_size=train_batch_size)
    print('\nGenerating dev dataset ...')
    dev_generator = generate_all_contours(dev_ctrs,
                                          shuffle=False,
                                          crop_size=crop_size,
                                          chunk_size=dev_batch_size)
    dev_batch = batch_data(dev_generator, batch_size=dev_batch_size)
    
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

    epochs = 10
    mini_batch_size = 1
    num_steps_per_epoch = len(train_ctrs) / train_batch_size
    max_iter = (len(train_ctrs) / mini_batch_size) * epochs
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    for e in range(epochs):
        print('\nMain Epoch {:d}\n'.format(e+1))
        step = 0
        for img_train, mask_train in train_batch: # each batch is 1000 files
            print('\nEpoch {:d} - Step {:d}/{:d}\n'.format(e+1, step+1,
                                                    num_steps_per_epoch))
            print('\nLearning rate: {:6f}\n'.format(lrate))
            # each augmented images and masks is a mini batch of files
            aug_images = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
            aug_masks = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
            train_dataset = izip(aug_images, aug_masks)
            for iteration in range(len(img_train)/mini_batch_size):
                img, mask = next(train_dataset)
                loss = model.train_on_batch(img, mask, class_weight=None)
                curr_iter += 1
                lrate = lr_poly_decay(model, base_lr, curr_iter,
                                      max_iter, power=0.5)
            print('Train result {:s}:\n{:s}'.format(model.metrics_names, loss))
            step += 1
            if step >= num_steps_per_epoch:
                print('\nEvaluating dev set ...')
                step = 0
                results = []
                for img_dev, mask_dev in dev_batch: # batch of 100 files
                    res = model.evaluate(img_dev, mask_dev, verbose=0,
                                         batch_size=len(img_dev))
                    results.append(res)
                    step += 1
                    if step >= len(dev_ctrs) / dev_batch_size:
                        break
                results = np.asarray(results)
                results = np.mean(results, axis=0).round(decimals=10)
                print('Dev result {:s}:\n{:s}'.format(model.metrics_names, results))
                save_file = '_'.join(['lvsc', contour_type,
                                      'epoch', str(e+1)]) + '.h5'
                if not os.path.exists('model_logs'):
                    os.makedirs('model_logs')
                save_path = os.path.join('model_logs', save_file)
                print('\nSaving model weights to {:s}'.format(save_path))
                model.save_weights(save_path)
                break



