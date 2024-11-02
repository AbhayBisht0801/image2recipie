from Image2Recipie.config.configuration import ConfigurationManager
from Image2Recipie.components.prepare_model import PrepareModel
from Image2Recipie import logger

STAGE_NAME="Model Preperation stage"

class PrepareModelPipeline:
    def __init__(self):
        pass
    def main(self):

        config=ConfigurationManager()
        model_prepare_config=config.get_prepare_model_config()
        model_preperation=PrepareModel(config=model_prepare_config)
        model_preperation.prepare_model()

if __name__=='__main__':
    try:
        logger.info(f'>>>>stage {STAGE_NAME} started <<<<')
        obj=PrepareModelPipeline()
        obj.main()
        logger.info(f'>>>>stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        logger.exception(e)
        raise e