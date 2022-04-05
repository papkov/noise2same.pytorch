#!/bin/bash

# Move to the working directory
if ${PWD##*/} == "scripts" ; then
    cd ..
fi

for image in ./data/ssi/*.png; do
  [ -e "$image" ] || continue
  image_name="${image##*/}"
  echo "$image_name"
  python train.py +experiment=ssi project=noise2same-ssi-paper data.input_name="$image_name" device=0 model.only_masked=True training.monitor=bsp_mse
done
