import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    return


if __name__ == "__main__":
    main()
