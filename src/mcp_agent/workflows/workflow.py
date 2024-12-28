from abc import ABC, abstractmethod


class BaseWorkflow(ABC):
    @abstractmethod
    def run(self):
        pass
