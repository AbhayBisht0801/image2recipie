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