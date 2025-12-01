from abc import ABC, abstractmethod
from typing import Sequence

from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from .schemas import BaseResponse


class BaseMCPTool(ABC):
    def __init__(self):
        pass

    def get_response_model(self) -> type[BaseResponse]:
        """Get the response model for this tool's endpoints.
        Can be overridden by subclasses to provide custom response models.
        """
        return BaseResponse

    @abstractmethod
    async def list_tools(self) -> list[Tool]:
        """List the tools provided by this module."""

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        arguments: dict,
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call a tool by name with given arguments."""

    @abstractmethod
    def get_route_config(self) -> list[dict]:
        """Get route configurations for this tool.

        Returns:
            list[dict]: List of configurations containing 'endpoint', 'params_class', etc.
        """
