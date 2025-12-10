from typing import List
from pydantic import BaseModel, Field


class ResumeAnalysisModel(BaseModel):
    score: int = Field(ge=0, le=100)
    justification: str
    missing_keywords: List[str]
    suggested_bullets: List[str]
