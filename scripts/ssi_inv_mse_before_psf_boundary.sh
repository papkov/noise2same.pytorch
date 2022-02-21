#!/bin/bash

# Move to the working directory
if ${PWD##*/} == "scripts" ; then
    cd ..
fi

for image in ./data/ssi/*.png; do
  [ -e "$image" ] || continue
  image_name="${image##*/}"
  echo "$image_name"
  python train.py +experiment=ssi project=noise2same-ssi-cfg data.input_name="$image_name" model.lambda_bound=0.1 model.inv_mse_key=deconv model.regularization_key=deconv device=4
done
