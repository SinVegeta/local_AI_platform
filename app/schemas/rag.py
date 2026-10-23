from pydantic import BaseModel, Field


class RagIndexRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)
    chunk_size: int = Field(default=900, ge=100, le=4000)
    overlap: int = Field(default=120, ge=0, le=1000)


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    model: str = Field(min_length=1, max_length=200)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=8)
    think: bool | None = None


class RagCitation(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    excerpt: str
    score: float
    extraction_method: str | None = None
    engine: str | None = None


class RagIndexResponse(BaseModel):
    indexes: list[dict]


class RagQueryResponse(BaseModel):
    answer: str
    model: str
    citations: list[RagCitation]
    retrieved_count: int
    warnings: list[str] = []
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    model_load_ms: int | None = None
    prompt_eval_ms: int | None = None
    generation_eval_ms: int | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    ollama_total_ms: int | None = None
