python train.py +experiment=microtubules_generated project=noise2same-ssi-mt-gen \
       model.lambda_inv=0 model.lambda_inv_deconv=2 \
       model.regularization_key=deconv model.lambda_bound=0.1 \
       device=5