from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract Environment"""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    @property
    def action_space(self):
        pass

    @abstractmethod
    @property
    def observation_space(self):
        pass

    @abstractmethod
    @property
    def n_actions(self):
        pass

    @abstractmethod
    @property
    def state_size(self):
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def isdiscrete(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
