import logging

def configure_logger(logger_name, log_filepath, level):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_filepath, 'w')
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def configure_debug_logger(logger_name, log_filepath):
    return configure_logger(logger_name, log_filepath, logging.DEBUG)

def configure_info_logger(logger_name, log_filepath):
    return configure_logger(logger_name, log_filepath, logging.INFO)