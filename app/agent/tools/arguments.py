"""Reusable constraints for local business tools; exported schema matches runtime."""
from typing import Annotated
from pydantic import Field, StringConstraints

OrderId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
RefundReason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
ProductKeyword = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
KnowledgeQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
MemoryQuery = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]
SkillName = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
TopK = Annotated[int, Field(ge=1, le=5)]
