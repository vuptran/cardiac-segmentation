#!/usr/bin/env python2.7
'''renames Sunnybrook submission files to match IM-0001-XXXX,
where XXXX is 4 digit index corresponding to the number of dicom images
'''
import os, re, sys

root_folder = sys.argv[1]

for case in os.listdir(root_folder):
    case_path = os.path.join(root_folder, case)
    if os.path.isdir(case_path):
        d = {}
        dcm_files = [f for dirpath, dirnames, files in os.walk(case_path)
                       for f in sorted(files) if '.dcm' in f]
        for idx, dcm in enumerate(dcm_files):
            match = re.search(r'IM-0001-(\d{4})', dcm)
            d[match.group(1)] = '{:04d}'.format(idx+1)
        for dirpath, dirnames, files in os.walk(case_path):
            for f in sorted(files):
                match = re.search(r'IM-0001-(\d{4})', f)
                if match:
                    f_new = f.replace(match.group(1), d[match.group(1)])
                    os.rename(os.path.join(dirpath, f), os.path.join(dirpath, f_new))

