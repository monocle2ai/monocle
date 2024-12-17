from abc import ABC, abstractmethod

class TaskProcessor(ABC):

    @abstractmethod
    def pre_task_processing(self, to_wrap, wrapped, instance, args, span):
        pass

    @abstractmethod
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, return_value, span):
        pass
