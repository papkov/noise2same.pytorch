# Noise2Same WIP

PyTorch reimplementation of [Noise2Same](https://github.com/divelab/Noise2Same).
Work in progress.

## Usage

Default configuration is located in `config/config.yaml`. 
Experiment configs `config/experiments` may override defaults.

To run an experiment for BSD68, execute
```bash
python train.py +experiment=bsd68
```
Four experiments from Noise2Same are supported: `bsd68`, `hanzi`, `imagenet`, `planaria`


## Results replication

We replicate the main results of Noise2Same (Table 3)

| Dataset             | Paper                 | Ours  |
|---------------------|-----------------------|-------|
| BSD68               | 27.95                 | 28.12 |
| HanZi               | 14.38                 | 14.83 |
| ImageNet            | 22.26                 | 22.81 |
| Planaria (C1/C2/C3) | 29.48 / 26.93 / 22.41 | WIP   |

