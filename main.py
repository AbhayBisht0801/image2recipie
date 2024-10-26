from Image2Recipie import logger
from Image2Recipie.pipeline.stage_01_data_ingestion import DataingestionPipeline
from Image2Recipie.pipeline.stage_02_preprocessing import PreprocessingPipeline


STAGE_NAME='Data Ingestion Stage'

try:
    logger.info(f'>>>>stage {STAGE_NAME} started <<<<')
    obj=DataingestionPipeline()
    obj.main()
    logger.info(f'>>>>stage {STAGE_NAME} completed <<<<')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME='Data Preprocessing Stage'

try:
    logger.info(f'>>>>stage {STAGE_NAME} started <<<<')
    obj=PreprocessingPipeline()
    obj.main()
    logger.info(f'>>>>stage {STAGE_NAME} completed <<<<')
except Exception as e:
    logger.exception(e)
    raise e