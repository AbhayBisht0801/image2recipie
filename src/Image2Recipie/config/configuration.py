from Image2Recipie.constants import *
from Image2Recipie.utils.common import read_yaml,create_directories
from Image2Recipie.entity.config_entity import DataIngestionConfig,DataPreprocessConfig
class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH

    ):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    def get_data_preprocessing_config(self)->DataPreprocessConfig:
        config=self.config
        create_directories([config.preprocessing.root_dir])
        create_directories([config.preprocessing.input_preprocess])
        create_directories([config.preprocessing.output_preprocess])
        data_preprocess_config=DataPreprocessConfig(
            root_dir=config.preprocessing.root_dir,
            input_preprocess=config.preprocessing.input_preprocess,
            image_input=config.preprocessing.image_input,
            text_input=config.preprocessing.text_input,
            output_preprocess=config.preprocessing.output_preprocess,
            unzip_dir=config.data_ingestion.unzip_dir
        )
        return data_preprocess_config