"""
Shared — Document Schema
Used by all services to ensure consistent field names across the pipeline.
Import this wherever you need a typed document representation.
"""
from pydantic import BaseModel
from typing import Optional


class Document(BaseModel):
    doc_id: str
    title: str
    text: str
    abstract: Optional[str] = None
    authors: Optional[str] = None
    published: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    full_text: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str          # "{doc_id}_chunk_{index}"
    doc_id: str
    text: str
    chunk_index: int
    metadata: dict = {}
