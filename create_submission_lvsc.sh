#!/usr/bin/env bash

echo "Creating LVSC submission files ..."
python submit_lvsc.py $1
zip -rq LVSC_data/Validation_auto_contours.zip LVSC_data/Validation_auto_contours
echo "All Done."
