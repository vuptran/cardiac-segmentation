#!/usr/bin/env python2.7

import dicom, sys, os, cv2
import numpy as np

from fcn_model import fcn_model
from helpers import center_crop, reshape


LVSC_ROOT_PATH = 'LVSC_data'

VALIDATION_PATH = os.path.join(LVSC_ROOT_PATH, 'Validation')

def create_submission(dcm_list, data_path):
    crop_size = 200
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2

    oweights = 'weights/lvsc_o.h5'
    iweights = 'weights/lvsc_i.h5'
    omodel = fcn_model(input_shape, num_classes, weights=oweights)
    imodel = fcn_model(input_shape, num_classes, weights=iweights)

    images = np.zeros((len(dcm_list), crop_size, crop_size, 1))
    for idx, dcm_path in enumerate(dcm_list):
        img = read_dicom(dcm_path)
        img = center_crop(img, crop_size=crop_size)
        images[idx] = img
    opred_masks = omodel.predict(images, batch_size=32, verbose=1)
    ipred_masks = imodel.predict(images, batch_size=32, verbose=1)

    save_dir = data_path + '_auto_contours'
    prefix = 'MYFCN_' # change prefix to your unique initials
    for idx, dcm_path in enumerate(dcm_list):
        img = read_dicom(dcm_path)
        h, w, d = img.shape
        otmp = reshape(opred_masks[idx], to_shape=(h, w, d))
        otmp = np.where(otmp > 0.5, 255, 0).astype('uint8')
        itmp = reshape(ipred_masks[idx], to_shape=(h, w, d))
        itmp = np.where(itmp > 0.5, 255, 0).astype('uint8')
        assert img.shape == otmp.shape, 'Prediction does not match shape'
        assert img.shape == itmp.shape, 'Prediction does not match shape'
        tmp = otmp - itmp
        tmp = np.squeeze(tmp, axis=(2,))
        sub_dir = dcm_path[dcm_path.find('CAP_'):dcm_path.rfind('DET')]
        filename = prefix + dcm_path[dcm_path.rfind('DET'):].replace('.dcm', '.png')
        full_path = os.path.join(save_dir, sub_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        cv2.imwrite(os.path.join(full_path, filename), tmp)
        in_ = cv2.imread(os.path.join(full_path, filename), cv2.IMREAD_GRAYSCALE)
        if not np.allclose(in_, tmp):
            raise AssertionError('File read error: {:s}'.format(os.path.join(full_path, filename)))
        

def get_all_dicoms(data_path):
    dcm_list = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(data_path)
                for f in files if 'SA' in f]
    print('Number of examples: {:d}'.format(len(dcm_list)))
    
    return dcm_list


def read_dicom(dcm_path):
    f = dicom.read_file(dcm_path)
    img = f.pixel_array.astype('int')
    if img.ndim < 3:
        img = img[..., np.newaxis]

    return img
    

if __name__== '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python %s <gpu_id>' % sys.argv[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

    print('Processing Validation set ...')
    val_dicoms = get_all_dicoms(VALIDATION_PATH)
    create_submission(val_dicoms, VALIDATION_PATH)
    print('All done.')

