import os
import logging
from functools import wraps


def initialize_logger():
    """Initializes and returns a logger."""

    logger = logging.getLogger("execution_guard")
    logger.handlers.clear()  # Clear existing handlers to avoid duplicates

    formatter = logging.Formatter(
        f"[%(asctime)s]  %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(f'execution_guard.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # if global_rank == 0:
    #     # Add a console handler for the main process
    #     ch = logging.StreamHandler()
    #     ch.setLevel(logging.DEBUG)
    #     ch.setFormatter(formatter)
    #     logger.addHandler(ch)

    logger.setLevel(logging.DEBUG)
    return logger

def execution_guard(func, force_overwrite=False):
    """Decorator to execute the function only if 'finished.lock' is absent in the current directory."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = initialize_logger()
        finished_lock = os.path.join(os.getcwd(), "finished.lock")
        if os.path.exists(finished_lock) and not force_overwrite:
            logger.info("Operation aborted. 'finished.lock' file found in the current directory.")
            return None
        else:
            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".log"):
                    try:
                        os.remove(filename)
                        logger.info(f"Deleted log file: {filename}")
                    except Exception as e:
                        logger.exception(f"Failed to delete log file: {filename}")
        try:
            logger.info(f"Executing function: {func.__name__}")
            func(*args, **kwargs)
            with open(finished_lock, 'w') as lock_file:
                lock_file.write("finished")
        except Exception as e:
            logger.exception("Exception occurred during function execution.")
    return wrapper

# Example usage:
#
# @execution_guard
# def function_name(logger, *args, **kwargs):
#     logger.info("Function is running with args: %s, kwargs: %s", args, kwargs)
#
# function_name("example_arg", key="value")