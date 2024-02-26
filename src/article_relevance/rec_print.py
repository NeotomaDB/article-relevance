from .logs import get_logger

def rel_print(log, header = None, verbose = False):
    """_Log printer with option for headers._

    Args:
        log (_str_): _A valid f-string or string input._
        header (_str_, optional): _Optional header for the log._. Defaults to None.
        verbose (bool, optional): _Should the element be logged?_. Defaults to False.
    """
    if verbose:
        logger = get_logger(__name__)
        if header:
            logger.info(header)
        logger.info(log)
    return True
