#!/bin/bash

# Move to the working directory
if ${PWD##*/} == "scripts" ; then
    cd ..
fi

for image in ./data/ssi/*.png; do
  [ -e "$image" ] || continue
  image_name="${image##*/}"
  echo "$image_name"
  python train.py +experiment=ssi project=noise2same-ssi-cfg data.input_name="$image_name" device=4 model.lambda_bound=0.1
done
