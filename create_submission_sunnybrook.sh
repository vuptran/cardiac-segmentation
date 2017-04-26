#!/usr/bin/env bash

echo "Creating Sunnybrook submission files ..."
python submit_sunnybrook.py i $1
python submit_sunnybrook.py o $1
python rename_sunnybrook.py Sunnybrook_val_submission
python rename_sunnybrook.py Sunnybrook_online_submission
echo "All Done."
