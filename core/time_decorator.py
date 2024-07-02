from time import time

from core.logger import logger


def timing(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        logger.info(f'> {func.__name__} starts')
        result = func(*args, **kwargs)
        t2 = time()
        exec_time = spent_time(t1, t2)
        logger.info(f'> {func.__name__} executed in {exec_time}')
        return result
    return wrapper


def spent_time(start_time, end_time):
    minutes = (end_time - start_time) // 60
    seconds = (end_time - start_time) - (minutes * 60)

    return ' {:.0f} min {:.0f} sec'.format(minutes, seconds)