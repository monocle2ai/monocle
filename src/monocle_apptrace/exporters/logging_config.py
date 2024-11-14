import logging

# Configure the global logger
logging.basicConfig(
    level=logging.NOTSET,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../tests/logger.log")
    ]
)
logger = logging.getLogger("global_logger")
