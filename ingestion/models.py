from pydantic import BaseModel


class Document(BaseModel):
    id: str
    title: str
    url: str
    source: str
    path: str
    content: str


class Chunk(BaseModel):
    id: str
    document_id: str

    title: str
    section: str | None

    content: str

    url: str
    source: str

    chunk_index: int