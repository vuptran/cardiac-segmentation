#!/usr/bin/env python2.7

import re, sys, os
import shutil, cv2
import numpy as np

from train_rvsc import map_all_contours, read_contour
from fcn_model import fcn_model
from helpers import center_crop, reshape


RVSC_ROOT_PATH = 'RVSC_data'

TEST01_PATH = os.path.join(RVSC_ROOT_PATH, 'Test1Set')
TEST02_PATH = os.path.join(RVSC_ROOT_PATH, 'Test2Set')
TRAINING_PATH = os.path.join(RVSC_ROOT_PATH, 'TrainingSet')


def create_submission(contours, data_path):
    if contour_type == 'i':
        weights = 'weights/rvsc_i.h5'
    elif contour_type == 'o':
        weights = 'weights/rvsc_o.h5'
    else:
        sys.exit('\ncontour type "%s" not recognized\n' % contour_type)

    crop_size = 200
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, _ = read_contour(contour, data_path, return_mask=False)
        img = center_crop(img, crop_size=crop_size)
        images[idx] = img

    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    model = fcn_model(input_shape, num_classes, weights=weights)
    pred_masks = model.predict(images, batch_size=32, verbose=1)
    
    save_dir = data_path + '_auto_contours'
    num = 0
    for idx, ctr in enumerate(contours):
        img, _ = read_contour(ctr, data_path, return_mask=False)
        h, w, d = img.shape
        tmp = reshape(pred_masks[idx], to_shape=(h, w, d))
        assert img.shape == tmp.shape, 'Shape of prediction does not match'
        tmp = np.where(tmp > 0.5, 255, 0).astype('uint8')
        tmp2, coords, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not coords:
            print('No detection: %s' % ctr.ctr_path)
            coords = np.ones((1, 1, 1, 2), dtype='int')
        if len(coords) > 1:
            print('Multiple detections: %s' % ctr.ctr_path)

            #cv2.imwrite('multiple_dets/'+contour_type+'{:04d}.png'.format(idx), tmp)
            
            lengths = []
            for coord in coords:
                lengths.append(len(coord))
            coords = [coords[np.argmax(lengths)]]
            num += 1
        filename = 'P{:s}-{:s}-'.format(ctr.patient_no, ctr.img_no)+contour_type+'contour-auto.txt'
        full_path = os.path.join(save_dir, 'P{:s}'.format(ctr.patient_no)+'contours-auto')
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        with open(os.path.join(full_path, filename), 'w') as f:
            for coord in coords:
                coord = np.squeeze(coord, axis=(1,))
                coord = np.append(coord, coord[:1], axis=0)
                np.savetxt(f, coord, fmt='%i', delimiter=' ')
    
    print('Num of files with multiple detections: {:d}'.format(num))


if __name__== '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    print('Processing Training '+contour_type+' contours...')
    train_ctrs = map_all_contours(TRAINING_PATH, contour_type, shuffle=False)
    create_submission(train_ctrs, TRAINING_PATH)
    print('Processing Test1 '+contour_type+' contours...')
    test1_ctrs = map_all_contours(TEST01_PATH, contour_type, shuffle=False)
    create_submission(test1_ctrs, TEST01_PATH)
    print('Processing Test2 '+contour_type+' contours...')
    test2_ctrs = map_all_contours(TEST02_PATH, contour_type, shuffle=False)
    create_submission(test2_ctrs, TEST02_PATH)
    print('All done.')
    
