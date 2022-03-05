import os
import time
from bg_rl.utilities.logging import configure_debug_logger

TIMING_LOGGER = configure_debug_logger('timing_logger', os.path.join('demos', 'logs', 'timing.log'))

def time_func(f):

    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        TIMING_LOGGER.debug(f"{f.__name__} took {te-ts} seconds")
        return result
    
    return timed