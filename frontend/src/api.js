const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
  } catch {
    throw new Error("Could not reach the local backend. Is FastAPI running?");
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // Keep the status-specific fallback below for non-JSON failures.
  }
  if (!response.ok) {
    throw new Error(payload?.detail || `Local backend returned HTTP ${response.status}.`);
  }
  return payload;
}

export const getHealth = () => request("/api/v1/health");
export const getModels = () => request("/api/v1/models");
export const getTasks = () => request("/api/v1/tasks");
export const sendChat = (model, message, task = "chat") =>
  request("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify({ model, message, task }),
  });

export async function uploadDocument(file, config = {}) {
  let response;
  const body = new FormData();
  body.append("file", file);
  Object.entries(config).forEach(([key, value]) => body.append(key, String(value)));
  try { response = await fetch(`${API_BASE_URL}/api/v1/documents/upload`, { method: "POST", body }); }
  catch { throw new Error("Could not reach the local backend."); }
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(payload?.detail || `Upload failed with HTTP ${response.status}.`);
  return payload;
}
export const getDocuments = () => request("/api/v1/documents");
export const getDocument = (id) => request(`/api/v1/documents/${id}`);
export const getDocumentText = (id) => request(`/api/v1/documents/${id}/text`);
export const deleteDocument = (id) => request(`/api/v1/documents/${id}`, { method: "DELETE" });
export const getRagIndexes = () => request("/api/v1/rag/indexes");
export const indexRagDocuments = (document_ids) => request("/api/v1/rag/index", { method: "POST", body: JSON.stringify({ document_ids }) });
export const deleteRagIndex = (id) => request(`/api/v1/rag/indexes/${id}`, { method: "DELETE" });
export const queryRag = (payload) => request("/api/v1/rag/query", { method: "POST", body: JSON.stringify(payload) });

