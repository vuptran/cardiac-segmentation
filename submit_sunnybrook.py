#!/usr/bin/env python2.7

import re, sys, os
import shutil, cv2
import numpy as np

from train_sunnybrook import read_contour, map_all_contours, export_all_contours
from fcn_model import fcn_model
from helpers import reshape, get_SAX_SERIES

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = 'Sunnybrook_data'
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart2',
                   'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_validation')
ONLINE_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart1',
                   'OnlineDataContours')
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'challenge_online')


def create_submission(contours, data_path):
    if contour_type == 'i':
        weights = 'weights/sunnybrook_i.h5'
    elif contour_type == 'o':
        weights = 'weights/sunnybrook_o.h5'
    else:
        sys.exit('\ncontour type "%s" not recognized\n' % contour_type)

    crop_size = 100
    images, masks = export_all_contours(contours, data_path, crop_size)
    
    input_shape = (crop_size, crop_size, 1)
    num_classes = 2
    model = fcn_model(input_shape, num_classes, weights=weights)

    pred_masks = model.predict(images, batch_size=32, verbose=1)
    num = 0
    for idx, ctr in enumerate(contours):
        img, mask = read_contour(ctr, data_path)
        h, w, d = img.shape
        tmp = reshape(pred_masks[idx], to_shape=(h, w, d))
        assert img.shape == tmp.shape, 'Shape of prediction does not match'
        tmp = np.where(tmp > 0.5, 255, 0).astype('uint8')
        tmp2, coords, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not coords:
            print('\nNo detection: %s' % ctr.ctr_path)
            coords = np.ones((1, 1, 1, 2), dtype='int')
        if len(coords) > 1:
            print('\nMultiple detections: %s' % ctr.ctr_path)
            
            #cv2.imwrite('multiple_dets/'+contour_type+'{:04d}.png'.format(idx), tmp)
            
            lengths = []
            for coord in coords:
                lengths.append(len(coord))
            coords = [coords[np.argmax(lengths)]]
            num += 1

        man_filename = ctr.ctr_path[ctr.ctr_path.rfind('/')+1:]
        auto_filename = man_filename.replace('manual', 'auto')
        img_filename = re.sub(r'-[io]contour-manual.txt', '.dcm', man_filename)
        man_full_path = os.path.join(save_dir, ctr.case, 'contours-manual', 'IRCCI-expert')
        auto_full_path = os.path.join(save_dir, ctr.case, 'contours-auto', 'FCN')
        img_full_path = os.path.join(save_dir, ctr.case, 'DICOM')
        dcm = 'IM-%s-%04d.dcm' % (SAX_SERIES[ctr.case], ctr.img_no)
        dcm_path = os.path.join(data_path, ctr.case, dcm)
        for dirpath in [man_full_path, auto_full_path, img_full_path]:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            if 'manual' in dirpath:
                src = ctr.ctr_path
                dst = os.path.join(dirpath, man_filename)
                shutil.copyfile(src, dst)
            elif 'DICOM' in dirpath:
                src = dcm_path
                dst = os.path.join(dirpath, img_filename)
                shutil.copyfile(src, dst)
            else:
                dst = os.path.join(auto_full_path, auto_filename)
                with open(dst, 'w') as f:
                    for coord in coords:
                        coord = np.squeeze(coord, axis=(1,))
                        coord = np.append(coord, coord[:1], axis=0)
                        np.savetxt(f, coord, fmt='%i', delimiter=' ')
    
    print('\nNumber of multiple detections: {:d}'.format(num))


if __name__== '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    save_dir = 'Sunnybrook_val_submission'
    print('\nProcessing val '+contour_type+' contours...')
    val_ctrs = map_all_contours(VAL_CONTOUR_PATH, contour_type, shuffle=False)
    create_submission(val_ctrs, VAL_IMG_PATH)

    save_dir = 'Sunnybrook_online_submission'
    print('\nProcessing online '+contour_type+' contours...')
    online_ctrs = map_all_contours(ONLINE_CONTOUR_PATH, contour_type, shuffle=False)
    create_submission(online_ctrs, ONLINE_IMG_PATH)
    print('\nAll done.')

