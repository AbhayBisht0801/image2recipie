import os
import urllib.request as request 
import gdown
from Image2Recipie import logger
from Image2Recipie.utils.common import get_size
import zipfile
from Image2Recipie.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config

    def download_file(self)->str:
        try:
            dataset_url=self.config.source_URL
            zip_download_dir=self.config.local_data_file
            os.makedirs('artifacts/data_ingestion',exist_ok=True)
            logger.info(f'Downloading data from {dataset_url} into file {zip_download_dir}')
            file_id=dataset_url.split('/')[-2]
            prefix='http://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)
            logger.info(f'Download data from {dataset_url} into file {zip_download_dir}')
        except Exception as e:
            raise e
    def extract_zip_file(self):
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)