from abc import ABC, abstractmethod
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class ExportTaskProcessor(ABC):

    @abstractmethod
    def start(self):
        return

    @abstractmethod
    def stop(self):
        return

    @abstractmethod
    def queue_task(self, async_task: Callable[[Callable, any], any] = None, args: any = None):
        return