from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """Base response model for all tool endpoints."""

    success: bool = True
    data: T | None = None
    error: str | None = None

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "success": True,
                "data": {},
                "error": None,
            },
        }
