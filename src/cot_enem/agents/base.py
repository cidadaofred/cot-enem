"""Base contract for evolutionary agents."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from cot_enem.dataset.schema import NormalizedQuestion

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class EvolutionAgent(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    def evolve(self, question: NormalizedQuestion, value: InputT) -> OutputT:
        raise NotImplementedError
