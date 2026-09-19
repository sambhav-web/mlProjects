import logging
import os
from datetime import datetime
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
Logs_Path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(Logs_Path,exist_ok=True)
LOG_FILE_PATH=os.path.join(Logs_Path,LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s ], %(lineno)d %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
if __name__=='__main__':
    logging.info('Logging has started...')