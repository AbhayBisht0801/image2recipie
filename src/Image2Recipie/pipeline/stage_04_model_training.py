from Image2Recipie.config.configuration import ConfigurationManager
from Image2Recipie.components.model_training import ModelTraining
from Image2Recipie import logger

STAGE_NAME="Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):

        config=ConfigurationManager()
        model_prepare_config=config.get_prepare_model_config()
        model_preperation=ModelTraining(config=model_prepare_config)
        model_preperation.get_model()
        model_preperation.model_training()

if __name__=='__main__':
    try:
        logger.info(f'>>>>stage {STAGE_NAME} started <<<<')
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f'>>>>stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        logger.exception(e)
        raise e