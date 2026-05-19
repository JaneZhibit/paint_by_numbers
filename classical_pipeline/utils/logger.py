import time
import logging

# Настройка базового логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PaintByNumbers")

class Timer:
    """Контекстный менеджер для замера времени этапов."""
    def __init__(self, stage_name: str, timings_dict: dict):
        self.stage_name = stage_name
        self.timings_dict = timings_dict
        self.start_time = None

    def __enter__(self):
        logger.info(f"Начало этапа: {self.stage_name}...")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.timings_dict[self.stage_name] = elapsed
        if exc_type is None:
            logger.info(f"Этап '{self.stage_name}' завершен за {elapsed:.3f} сек.")
        else:
            logger.error(f"Ошибка на этапе '{self.stage_name}': {exc_val}")
            