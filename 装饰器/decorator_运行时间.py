import logging
from functools import wraps
import time

def cost_time(func):
    @wraps(func)
    def log(*args, **kwargs):

        logger = logging.getLogger(func.__name__)
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        fm = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        sh.setFormatter(fm)
        logger.addHandler(sh)

        t_start = time.time()
        func(*args, **kwargs)
        t_end = time.time()
        t = round(t_end - t_start,3)
        logger.info(f'运行时间:{t}秒')


    return log
