from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    LLAMA_3_3_70_B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8_B = "llama-3.1-8b-instant"


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.LLAMA_3_1_8_B)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int
