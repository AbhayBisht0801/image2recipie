from Image2Recipie.config.configuration import ConfigurationManager
from Image2Recipie.components.preprocessing import DataPreprocess
from Image2Recipie import logger
STAGE_NAME='Preprocessing'
class PreprocessingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config=ConfigurationManager()
            data_preprocessing_config=config.get_data_preprocessing_config()
            data_preprocessing=DataPreprocess(config=data_preprocessing_config)
            data_preprocessing.input_preprocess()
            data_preprocessing.output_preprcess()
            
        except Exception as e:
            raise e
if __name__=='__main__':
    try:
        logger.info(f'>>>>stage {STAGE_NAME} started <<<<')
        obj=PreprocessingPipeline()
        obj.main()
        logger.info(f'>>>>stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        logger.exception(e)
        raise e