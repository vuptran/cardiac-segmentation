#!/usr/bin/env bash

echo "Creating RVSC submission files ..."
python submit_rvsc.py i $1
python submit_rvsc.py o $1
zip -rq RVSC_data/Test1Set_auto_contours.zip RVSC_data/Test1Set_auto_contours
zip -rq RVSC_data/Test2Set_auto_contours.zip RVSC_data/Test2Set_auto_contours
echo "All Done."
