from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseConfig(ABC, BaseModel):
    @abstractmethod
    def get_attributes(self) -> dict[str, str|int|float|bool]:
        pass
