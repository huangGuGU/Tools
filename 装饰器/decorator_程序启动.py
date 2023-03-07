import logging
from functools import wraps
import time


def logit(func):
    @wraps(func)
    def log(*args, **kwargs):

        logger = logging.getLogger(func.__name__)
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        # fm = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        # sh.setFormatter(fm)
        logger.addHandler(sh)
        logger.info(f'{func.__name__}  启动 ')
        func(*args, **kwargs)
        time.sleep(0.1)
        logger.info(f'{func.__name__}  完成')
        # return func(*args, **kwargs)


    return log