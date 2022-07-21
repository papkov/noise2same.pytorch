#!/bin/bash

# Move to the working directory
if ${PWD##*/} == "scripts" ; then
    cd ..
fi

for image in ./data/ssi/*.png; do
  [ -e "$image" ] || continue
  image_name="${image##*/}"
  echo "$image_name"
  python train.py +experiment=ssi +backbone=unet project=noise2same-ssi-paper \
                  data.input_name="$image_name" model.lambda_inv=2 model.lambda_inv_deconv=0 \
                  model.lambda_bound=0.1  \
                  device=3  # change device accordingly
done