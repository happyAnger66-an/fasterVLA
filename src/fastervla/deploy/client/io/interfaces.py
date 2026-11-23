"""Embodiment-agnostic robot IO interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastervla.core.types import RobotCommand, RobotState


class StateSource(ABC):
    @abstractmethod
    def read(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class CommandSink(ABC):
    @abstractmethod
    def send(self, cmd: RobotCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class ImageSource(ABC):
    @abstractmethod
    def read(self) -> dict[str, bytes]:
        raise NotImplementedError

    @abstractmethod
    def drop_frame(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class RobotPlatform(ABC):
    """Bundles state / command / images for one embodiment."""

    @property
    @abstractmethod
    def state(self) -> StateSource:
        raise NotImplementedError

    @property
    @abstractmethod
    def command(self) -> CommandSink:
        raise NotImplementedError

    @property
    @abstractmethod
    def images(self) -> ImageSource:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
