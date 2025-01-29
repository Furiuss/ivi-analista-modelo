import logging
from functools import wraps
import time
from typing import Callable, Any


def setup_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.disabled = True
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_execution_time(logger: logging.Logger):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"{func.__name__} executed in {execution_time:.2f} seconds"
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise

        return wrapper

    return decorator