from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class  DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file: Path
    unzip_dir:Path
@dataclass(frozen=True)
class  DataPreprocessConfig:
    root_dir:Path
    input_preprocess:Path
    image_input:Path
    text_input:Path
    output_preprocess:Path
    unzip_dir:Path
@dataclass(frozen=True)
class  Model_prepare_Config:
    root_dir:Path
    training_model:Path
    text_feature:Path
    params_image_size:list
    params_learning_rate:float
    params_weights:str
    params_include_top:bool
@dataclass(frozen=True)
class  Model_Training_Config:
    root_dir:Path
    training_model:Path
    params_batch_size:int
    output_preprocess:Path
    input_preprocess:Path
    image_input:Path
    text_input:Path
    params_epochs:int
    trained_model_path:Path