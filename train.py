import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from data_module import CustomDataModule
from model_module import CustomModel

# Указываем путь к директории configs
@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    # Получаем параметры из YAML файлов
    data_module = CustomDataModule(cfg.data.path, cfg.data.batch_size)
    model = CustomModel(cfg.model.model_params)
    trainer = Trainer(max_epochs=cfg.train.max_epochs)
    trainer.fit(model, datamodule=data_module)
    #data_module = CustomDataModule("./data/train.csv", 32)
    #model = CustomModel(5)
    #trainer = Trainer(max_epochs=10)
    #trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()