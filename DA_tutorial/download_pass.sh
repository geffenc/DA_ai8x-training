#!/bin/bash

# download files
echo "downloading dataset tar files"
# This will only install partition 0. Change to 0 1 2 ... if you want more.
# Each partition is 150K images and ~ 17GB
for PART in 0 
do
   echo "download part" $PART
   curl  https://zenodo.org/record/6615455/files/PASS.${PART}.tar --output PASS.${PART}.tar
done

# extract dataset
## will create dataset with images in PASS_dataset/dummy_folder/img-hash.jpeg
for file in *.tar; do tar -xf "$file"; done

