"""HTTP routes for the local AI platform."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.providers.base import (
    ProviderConnectionError,
    ProviderError,
    ProviderModelIncompatibleError,
    ProviderModelNotFoundError,
    TaskPipelineUnavailableError,
)
from app.providers.ollama import OllamaProvider
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.schemas.models import ModelsResponse, TaskName, TaskInfo
from app.schemas.documents import DocumentDetail, DocumentSummary, DocumentTextResponse, OCRConfig
from app.schemas.rag import RagIndexRequest, RagIndexResponse, RagQueryRequest, RagQueryResponse
from app.services.documents import DocumentBusyError, DocumentInputError, DocumentService
from app.services.rag_service import RagInputError, RagService
from app.services.rag_store import RagStore
from app.services.platform import PlatformService

router = APIRouter()
_document_service: DocumentService | None = None
_rag_service: RagService | None = None


def get_documents(settings: Settings = Depends(get_settings)) -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService(settings)
    return _document_service


def get_service(settings: Settings = Depends(get_settings)) -> PlatformService:
    return PlatformService(OllamaProvider(settings))


def get_rag(settings: Settings = Depends(get_settings), documents: DocumentService = Depends(get_documents)) -> RagService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagService(settings, document_store=documents.store)
    return _rag_service


@router.get("/tasks", response_model=list[TaskInfo])
async def tasks(service: PlatformService = Depends(get_service)) -> list[TaskInfo]:
    return service.list_tasks()


@router.get("/health", response_model=HealthResponse)
async def health(service: PlatformService = Depends(get_service)) -> HealthResponse:
    ollama_connected = await service.check_provider()
    return HealthResponse(status="ok", ollama_connected=ollama_connected)


@router.get("/models", response_model=ModelsResponse)
async def models(task: TaskName | None = None, service: PlatformService = Depends(get_service)) -> ModelsResponse:
    try:
        return ModelsResponse(models=await service.list_models(task=task))
    except ProviderConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, service: PlatformService = Depends(get_service)) -> ChatResponse:
    try:
        return await service.chat(request)
    except ProviderModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderModelIncompatibleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TaskPipelineUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/documents/upload", response_model=DocumentDetail, status_code=202)
async def upload_document(request: Request, service: DocumentService = Depends(get_documents)) -> DocumentDetail:
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type or "boundary=" not in content_type:
            raise DocumentInputError("Upload must use multipart/form-data with a file field")
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
        content_length = int(request.headers.get("content-length", "0") or 0)
        if content_length and content_length > 26 * 1024 * 1024:
            raise DocumentInputError("Upload exceeds the local size limit")
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > 26 * 1024 * 1024:
                raise DocumentInputError("Upload exceeds the local size limit")
            chunks.append(chunk)
        body = b"".join(chunks)
        marker = b"--" + boundary
        filename = None
        data = None
        fields: dict[str, str] = {}
        for part in body.split(marker):
            if b"Content-Disposition:" not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end < 0:
                continue
            headers = part[:header_end].decode("latin-1", errors="ignore")
            content = part[header_end + 4:]
            if content.endswith(b"\r\n"):
                content = content[:-2]
            import re
            match = re.search(r'name="([^"]+)"', headers)
            if match and match.group(1) == "file":
                filename_match = re.search(r'filename="([^"]*)"', headers)
                filename = filename_match.group(1) if filename_match else "document"
                data = content
            elif match:
                fields[match.group(1)] = content.decode("utf-8", errors="replace")
        if filename is None or data is None:
            raise DocumentInputError("The multipart request does not contain a file")
        config = OCRConfig(
            language=fields.get("language", "en"),
            primary_engine=fields.get("primary_engine", "paddleocr"),
            enable_fallback=fields.get("enable_fallback", "true").lower() == "true",
        )
        return await service.upload_bytes(filename, data, config)
    except DocumentInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(service: DocumentService = Depends(get_documents)) -> list[DocumentSummary]:
    return service.store.list()


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, service: DocumentService = Depends(get_documents)) -> DocumentDetail:
    detail = service.store.get(document_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")
    return detail


@router.get("/documents/{document_id}/text", response_model=DocumentTextResponse)
async def get_document_text(document_id: str, service: DocumentService = Depends(get_documents)) -> DocumentTextResponse:
    detail = service.store.get(document_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentTextResponse(document_id=document_id, text="\n\n".join(page.text for page in detail.pages if page.text), pages=detail.pages)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, service: DocumentService = Depends(get_documents)) -> None:
    try:
        deleted = service.delete(document_id)
    except DocumentBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    if _rag_service:
        _rag_service.delete_index(document_id)
    else:
        RagStore().delete(document_id)


@router.post("/rag/index", response_model=RagIndexResponse)
async def index_documents(request: RagIndexRequest, service: RagService = Depends(get_rag)) -> RagIndexResponse:
    try:
        return await service.index(request.document_ids, request.chunk_size, request.overlap)
    except RagInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/rag/indexes", response_model=list[dict])
async def list_indexes(service: RagService = Depends(get_rag)) -> list[dict]:
    return service.indexes()


@router.get("/rag/indexes/{document_id}", response_model=dict)
async def get_index(document_id: str, service: RagService = Depends(get_rag)) -> dict:
    result = service.index_detail(document_id)
    if not result:
        raise HTTPException(status_code=404, detail="Index not found")
    return result


@router.delete("/rag/indexes/{document_id}", status_code=204)
async def delete_index(document_id: str, service: RagService = Depends(get_rag)) -> None:
    if not service.delete_index(document_id):
        raise HTTPException(status_code=404, detail="Index not found")


@router.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(request: RagQueryRequest, service: RagService = Depends(get_rag)) -> RagQueryResponse:
    try:
        return await service.query(request)
    except RagInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
