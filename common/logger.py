import logging
import sys
from loguru import logger


class InterceptHandler(logging.Handler):
    """Intercept standard library logging and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Use raw=True to prevent format string processing issues
        logger.opt(depth=depth, exception=record.exc_info, raw=True).log(level, record.getMessage() + "\n")

class HermesLogger:

    @classmethod
    def configure_loguru(
        cls,
        console_level: str = "INFO",

        file: str = None,
        error_file: str = None,
        file_level: str = "INFO",
        file_json: bool = False
    ):
        """
        Configure loguru to intercept standard logging and suppress noisy third-party libraries.
        """

        # Remove default loguru handler and add custom one
        logger.remove()

        def format_record(record) -> str:
            source = record["extra"].get("source", "")
            if source:
                return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |  <cyan>{extra[source]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}\n"
            else:
                return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}\n"

        logger.add(
            sys.stdout,
            level=console_level,
            format=format_record
        )

        if file:
            logger.add(
                file, 
                level=file_level, 
                rotation="00:00", # New file at 0:00 every day
                retention="15 days",
                serialize=file_json, # If set, format will be ignored
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
            )
        if error_file:
            logger.add(
                error_file, 
                level="ERROR", 
                rotation="00:00", # New file at 0:00 every day
                retention="15 days",
                serialize=file_json, # If set, format will be ignored
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
            )

        # Intercept standard library logging
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
        # Configure specific loggers to reduce noise
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.WARNING) 
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# python common/logger.py 
if __name__ == "__main__":
    HermesLogger.configure_loguru(file="logs/hermes.log")

    logger.info("Logger initialized")
    logger.bind(source='xxx').info("Logger initialized")
    logger.bind(source='xxx').info("{}")
    logger.bind(source='xxx').info("params: {}", {"w": 2})
