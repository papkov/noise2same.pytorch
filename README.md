# Noise2Same (PyTorch)

PyTorch reimplementation of [Noise2Same](https://github.com/divelab/Noise2Same).
Work in progress.

## Usage

Default configuration is located in `config/config.yaml`. 
Experiment configs `config/experiments` may override defaults.

### Training

To run an experiment for BSD68, execute
```bash
python train.py +experiment=bsd68
```
Four experiments from Noise2Same are supported: `bsd68`, `hanzi`, `imagenet`, `planaria`.

Training logs and model weights will be saved to `resuts/train/datetime`.

### Evaluation

To run evaluation for BSD68, execute
```bash
python evaluate.py +experiment=bsd68
```
By default, we assume the weights for the model to be in `weights/experiment.pth`
but you can specify the path by adding `+checkpoint=/path/to/checkpoint`.

Model's outputs and scores (RMSE, PSNR, SSIM for each image) will be saved to `resuts/evaluate/datetime`.


## Results replication

We replicate the main results of Noise2Same (Table 3)
[]()

| Dataset             | Ours (Noise2Self) | Noise2Same paper      | Ours (Noise2Same)  |  Weights |
|---------------------|-------------------|-----------------------|-------|----------|
| BSD68               | 26.73             | 27.95                 | 28.11 | [Drive](https://drive.google.com/file/d/1YTlHpL-C4JaRtfp8YUiXfzppX0Tgs4lC/view?usp=sharing)|
| HanZi               |                   | 14.38                 | 14.83 |[Drive](https://drive.google.com/file/d/1WHd_BUqlibrDERWwzs4ReSZu2s8Ya1Y2/view?usp=sharing)|
| ImageNet            |                   | 22.26                 | 22.81 |[Drive](https://drive.google.com/file/d/12Rxp30DmwmYBq6ZtgnPD-SMd9u1FeRER/view?usp=sharing)|
| Planaria (C1/C2/C3) |                   | 29.48 / 26.93 / 22.41 | 29.14 / 27.11 / 22.80 |[Drive](https://drive.google.com/file/d/17Yz_f8RNOu7nEztSOug_1PLQFKYYj_Mf/view?usp=sharing)|

