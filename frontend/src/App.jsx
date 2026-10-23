import { useCallback, useEffect, useMemo, useState } from "react";
import { deleteDocument, deleteRagIndex, getDocument, getDocuments, getHealth, getModels, getRagIndexes, getTasks, indexRagDocuments, queryRag, sendChat, uploadDocument } from "./api";

const taskIcons = { chat: "chat", coding: "code", ocr: "scan", embeddings: "layers" };

function Icon({ name, size = 18 }) {
  const paths = {
    chat: <><path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.8 8.8 0 0 1-3.7-.8L4 20l1.5-3.7A7.4 7.4 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z" /><path d="M8 11h.01M12 11h.01M16 11h.01" /></>,
    code: <><path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" /></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><path d="M8 12h8M12 8v8" /></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
    refresh: <><path d="M20 11a8 8 0 0 0-14.9-4L3 10" /><path d="M3 5v5h5M4 13a8 8 0 0 0 14.9 4L21 14" /><path d="M21 19v-5h-5" /></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>,
    trash: <><path d="M3 6h18M8 6V4h8v2M19 6l-1 15H6L5 6M10 11v6M14 11v6" /></>,
    cpu: <><rect x="5" y="5" width="14" height="14" rx="2" /><path d="M9 9h6v6H9zM9 1v4M15 1v4M9 19v4M15 19v4M19 9h4M19 14h4M1 9h4M1 14h4" /></>,
    spark: <><path d="m12 3-1.2 5.8L5 10l5.8 1.2L12 17l1.2-5.8L19 10l-5.8-1.2Z" /><path d="m19 17-.5 2.5L16 20l2.5.5L19 23l.5-2.5L22 20l-2.5-.5Z" /></>,
    check: <path d="m5 12 4 4L19 6" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function normalized(identifier) { return (identifier || "").replace(/:latest$/, ""); }
function modelLabel(model) { return model.display_name || model.name || model.identifier; }

function OCRWorkspace() {
  const [documents, setDocuments] = useState([]); const [selected, setSelected] = useState(null);
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const [engine, setEngine] = useState("tesseract");
  const reload = useCallback(async () => { try { setDocuments(await getDocuments()); } catch (e) { setError(e.message); } }, []);
  useEffect(() => { reload(); const timer = setInterval(reload, 3000); return () => clearInterval(timer); }, [reload]);
  useEffect(() => { if (!selected || !["queued", "processing"].includes(selected.status)) return; const timer = setInterval(async () => { try { const next = await getDocument(selected.document_id); setSelected(next); setDocuments(await getDocuments()); } catch (e) { setError(e.message); } }, 1500); return () => clearInterval(timer); }, [selected]);
  async function upload(event) { const file = event.target.files?.[0]; if (!file) return; setBusy(true); setError(""); try { const doc = await uploadDocument(file, { primary_engine: engine, language: "en", enable_fallback: engine === "paddleocr" }); setSelected(doc); await reload(); } catch (e) { setError(e.message); } finally { setBusy(false); event.target.value = ""; } }
  async function remove(id) { if (!window.confirm("Delete this local document and its extracted results?")) return; try { await deleteDocument(id); if (selected?.document_id === id) setSelected(null); await reload(); } catch (e) { setError(e.message); } }
  const pageText = selected?.pages?.map((page) => `Page ${page.page_number}\n${page.text || "[No text extracted]"}`).join("\n\n") || "";
  return <section className="ocr-workspace"><div className="workspace-heading"><div><p className="eyebrow">LOCAL DOCUMENT EXTRACTION</p><h1>OCR workspace</h1><p className="subtitle">Process PDFs and images privately. Digital PDF text is extracted natively; scanned pages use PaddleOCR first and GLM-OCR only when needed.</p></div></div>
    <label className={`upload-dropzone ${busy ? "disabled" : ""}`}><span className="upload-icon">↑</span><strong>{busy ? "Uploading…" : "Drop a document here or choose a file"}</strong><small>PDF, PNG, JPG, JPEG · maximum 25 MB · sequential local processing</small><input type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onChange={upload} disabled={busy} /></label>
    <div className="ocr-controls"><label htmlFor="ocr-engine">OCR engine<select id="ocr-engine" value={engine} onChange={(e) => setEngine(e.target.value)} disabled={busy}><option value="tesseract">Tesseract · CPU · eng</option><option value="glm-ocr">GLM-OCR · local Ollama vision</option><option value="paddleocr">PaddleOCR · PP-OCRv5 worker</option></select></label><span>Only the selected engine is used for this upload.</span></div>
    {error && <div className="alert error-alert"><span>!</span><div>{error}</div></div>}
    <div className="ocr-grid"><div className="document-list"><div className="panel-title"><strong>Local documents</strong><button className="text-button" onClick={reload}>Refresh</button></div>{!documents.length && <p className="muted">No documents uploaded yet.</p>}{documents.map((doc) => <button className={`document-item ${selected?.document_id === doc.document_id ? "selected" : ""}`} key={doc.document_id} onClick={async () => setSelected(await getDocument(doc.document_id))}><span><strong>{doc.original_filename}</strong><small>{doc.page_count} page{doc.page_count === 1 ? "" : "s"} · {doc.status}</small></span><span className="doc-delete" onClick={(e) => { e.stopPropagation(); remove(doc.document_id); }}>×</span></button>)}</div>
      <div className="document-results">{!selected ? <div className="muted">Select a document to inspect page-level results.</div> : <><div className="panel-title"><div><strong>{selected.original_filename}</strong><small>{selected.status} · {selected.completed_pages}/{selected.page_count} pages complete</small></div><button className="text-button" onClick={() => navigator.clipboard?.writeText(pageText)}>Copy text</button></div><div className="result-meta">{selected.pages?.map((page) => <article className="page-result" key={page.page_number}><div className="page-header"><strong>Page {page.page_number}</strong><span>{page.engine || page.extraction_method || page.status}{page.confidence != null ? ` · ${(page.confidence * 100).toFixed(0)}% confidence` : ""}</span></div><pre>{page.text || page.error || "Waiting for extraction…"}</pre>{page.warnings?.map((warning) => <small className="warning-text" key={warning}>Warning: {warning}</small>)}</article>)}</div></>}</div></div>
  </section>;
}

function RAGWorkspace({ models }) {
  const [documents, setDocuments] = useState([]); const [indexes, setIndexes] = useState([]); const [selected, setSelected] = useState([]); const [question, setQuestion] = useState(""); const [answer, setAnswer] = useState(null); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const chatModels = models.filter((model) => model.installed && model.supported_tasks?.includes("chat"));
  const [model, setModel] = useState(""); const [think, setThink] = useState(false);
  const reload = useCallback(async () => { try { const [docs, currentIndexes] = await Promise.all([getDocuments(), getRagIndexes()]); setDocuments(docs); setIndexes(currentIndexes); } catch (e) { setError(e.message); } }, []);
  useEffect(() => { reload(); }, [reload]);
  useEffect(() => { if (!model && chatModels[0]) { const preferred = chatModels.find((item) => item.identifier.toLowerCase().startsWith("granite4.1")); setModel((preferred || chatModels[0]).identifier); } }, [chatModels, model]);
  const indexedIds = new Set(indexes.filter((item) => item.status === "indexed").map((item) => item.document_id));
  async function indexSelected() { setBusy(true); setError(""); try { await indexRagDocuments(selected); await reload(); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  async function ask(event) { event.preventDefault(); if (!question.trim() || !model || !selected.length || busy) return; setBusy(true); setError(""); try { setAnswer(await queryRag({ question: question.trim(), model, document_ids: selected, top_k: 5, ...(model.toLowerCase().startsWith("qwen3") ? { think } : {}) })); } catch (e) { setError(e.message); } finally { setBusy(false); } }
  function toggle(id) { setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]); }
  return <section className="workspace"><div className="workspace-heading"><div><p className="eyebrow">LOCAL RETRIEVAL</p><h1>Document Q&amp;A</h1><p className="subtitle">Index completed local documents, retrieve page-aware evidence, and ask a local model grounded questions.</p></div></div>
    {error && <div className="alert error-alert"><span>!</span><div>{error}</div></div>}
    <div className="model-panel"><div className="panel-title"><strong>Document collection</strong><button className="text-button" onClick={reload}>Refresh</button></div><p className="muted">Only completed OCR/text documents can be indexed. Indexing and retrieval stay on this machine.</p>{!documents.length && <p className="muted">No processed documents available.</p>}{documents.map((doc) => <label className="document-item" key={doc.document_id}><input type="checkbox" checked={selected.includes(doc.document_id)} onChange={() => toggle(doc.document_id)} disabled={!['completed','partially_completed'].includes(doc.status)} /><span><strong>{doc.original_filename}</strong><small>{doc.status} · {doc.page_count} page{doc.page_count === 1 ? '' : 's'} · {indexedIds.has(doc.document_id) ? 'Indexed' : 'Not indexed'}</small></span></label>)}<div className="model-toolbar"><button className="refresh-button" disabled={!selected.length || busy} onClick={indexSelected}>{busy ? 'Indexing…' : 'Index selected'}</button>{indexes.filter((item) => selected.includes(item.document_id)).map((item) => <button className="text-button" key={item.document_id} onClick={async () => { await deleteRagIndex(item.document_id); await reload(); }}>Remove index</button>)}</div></div>
    <div className="model-panel"><label htmlFor="rag-model">Answer model<select id="rag-model" value={model} onChange={(event) => setModel(event.target.value)}><option value="">Select an installed chat model</option>{chatModels.map((item) => <option key={item.identifier} value={item.identifier}>{modelLabel(item)} · {item.identifier}</option>)}</select></label>{model.toLowerCase().startsWith("qwen3") && <><label className="inline-option"><input type="checkbox" checked={think} onChange={(event) => setThink(event.target.checked)} /> Enable Qwen3 thinking (slower, more deliberate)</label><p className="model-help">For direct document explanations, Granite is recommended on this local Ollama version. Qwen3 may return incomplete reasoning output.</p></>}<form className="composer" onSubmit={ask}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={selected.length ? "Ask about the selected indexed documents…" : "Select indexed documents first…"} disabled={!selected.length || busy} rows="3" /><div className="composer-footer"><span>Answers include verified document and page citations.</span><button className="send-button" type="submit" disabled={!question.trim() || !model || !selected.length || busy}>{busy ? 'Querying' : 'Ask'} <Icon name="send" size={15} /></button></div></form></div>
    {answer && <div className="conversation has-messages"><div className="message-row assistant"><div className="avatar"><Icon name="spark" size={15} /></div><div className="message-body"><div className="message-label">{answer.model} · answered in {(answer.total_ms / 1000).toFixed(2)}s</div>{answer.answer?.trim() ? <div className="message-content rag-answer">{answer.answer}</div> : <div className="notice warning"><span>!</span><div>The local model returned no explanation. Please retry with Qwen3 thinking disabled.</div></div>}<small className="muted rag-timing">Retrieval: {(answer.retrieval_ms / 1000).toFixed(2)}s · Ollama generation: {(answer.generation_ms / 1000).toFixed(2)}s · {answer.retrieved_count} source{answer.retrieved_count === 1 ? '' : 's'}<br />Model load: {answer.model_load_ms == null ? "unavailable" : `${(answer.model_load_ms / 1000).toFixed(2)}s`} · Prompt: {answer.prompt_eval_ms == null ? "unavailable" : `${(answer.prompt_eval_ms / 1000).toFixed(2)}s`} · Generation: {answer.generation_eval_ms == null ? "unavailable" : `${(answer.generation_eval_ms / 1000).toFixed(2)}s`} · Speed: {answer.tokens_per_second == null ? "unavailable" : `${answer.tokens_per_second} tok/s`}</small>{answer.citations?.map((citation) => <div className="notice info" key={citation.chunk_id}><strong>{citation.document_name}, page {citation.page_number}</strong><br /><small>{citation.excerpt}</small><br /><small>Score {citation.score} · {citation.engine || citation.extraction_method || 'local extraction'}</small></div>)}</div></div></div>}
  </section>;
}

function App() {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState("chat");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [statusLoading, setStatusLoading] = useState(true);
  const [modelsError, setModelsError] = useState("");
  const [statusError, setStatusError] = useState("");
  const [chatError, setChatError] = useState("");

  const activeTask = useMemo(() => tasks.find((task) => task.id === selectedTask), [tasks, selectedTask]);
  const taskModels = useMemo(() => models.filter((model) => model.supported_tasks?.includes(selectedTask)), [models, selectedTask]);
  const installedModels = useMemo(() => taskModels.filter((model) => model.installed), [taskModels]);
  const activeModel = useMemo(() => models.find((model) => model.identifier === selectedModel), [models, selectedModel]);
  const preferredModel = activeTask ? models.find((model) => normalized(model.identifier) === normalized(activeTask.preferred_model)) : null;
  const preferredMissing = Boolean(activeTask && (!preferredModel || !preferredModel.installed));
  const isOnline = health?.status === "ok" && health?.ollama_connected === true;
  const taskCanChat = Boolean(activeTask?.pipeline_available && selectedModel);

  const refreshHealth = useCallback(async () => {
    setStatusLoading(true); setStatusError("");
    try { setHealth(await getHealth()); } catch (error) { setHealth(null); setStatusError(error.message); } finally { setStatusLoading(false); }
  }, []);

  const refreshModels = useCallback(async () => {
    setModelsLoading(true); setModelsError("");
    try {
      const payload = await getModels();
      setModels(Array.isArray(payload?.models) ? payload.models : []);
    } catch (error) { setModels([]); setSelectedModel(""); setModelsError(error.message); } finally { setModelsLoading(false); }
  }, []);

  useEffect(() => {
    Promise.all([getTasks(), refreshModels(), refreshHealth()]).then(([taskPayload]) => {
      const nextTasks = Array.isArray(taskPayload) ? taskPayload : [];
      setTasks(nextTasks);
      if (nextTasks.length && !nextTasks.some((task) => task.id === selectedTask)) setSelectedTask(nextTasks[0].id);
    }).catch((error) => setModelsError(error.message));
  // Initial loading is intentionally performed once.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!activeTask || modelsLoading) return;
    const preferredInstalled = taskModels.find((model) => normalized(model.identifier) === normalized(activeTask.preferred_model) && model.installed);
    const currentStillValid = installedModels.some((model) => model.identifier === selectedModel);
    if (!currentStillValid) setSelectedModel(preferredInstalled?.identifier || installedModels[0]?.identifier || "");
  }, [activeTask, modelsLoading, taskModels, installedModels, selectedModel]);

  function changeTask(taskId) {
    setSelectedTask(taskId); setMessages([]); setChatError(""); setDraft(""); setSelectedModel("");
  }

  async function handleSubmit(event) {
    event?.preventDefault();
    const message = draft.trim();
    if (!message || !selectedModel || !taskCanChat || isLoading) return;
    setChatError(""); setDraft(""); setMessages((current) => [...current, { role: "user", content: message }]); setIsLoading(true);
    try {
      const response = await sendChat(selectedModel, message, selectedTask);
      setMessages((current) => [...current, { role: "assistant", content: response.response, model: response.model }]);
    } catch (error) { setChatError(error.message); } finally { setIsLoading(false); }
  }

  function handleComposerKeyDown(event) { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); handleSubmit(event); } }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Icon name="spark" size={20} /></div><div><strong>Local AI</strong><span>Studio</span></div></div>
        <div className="nav-section-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Task navigation">
          {tasks.map((task) => <button className={`nav-item ${task.id === selectedTask ? "active" : ""}`} key={task.id} onClick={() => changeTask(task.id)}><Icon name={taskIcons[task.id] || "spark"} size={17} /> {task.display_name}<span className={task.pipeline_available ? "nav-kicker" : "planned"}>{task.pipeline_available ? "READY" : "SOON"}</span></button>)}
        </nav>
        <div className="sidebar-footer">
          <div className="runtime-card"><div className="runtime-heading"><span className={`status-dot ${isOnline ? "online" : "offline"}`} /> Local runtime</div><div className="runtime-value">{isOnline ? "Ollama connected" : statusLoading ? "Checking connection" : "Offline"}</div><button className="text-button" onClick={refreshHealth} disabled={statusLoading}><Icon name="refresh" size={14} /> Refresh status</button></div>
          <div className="privacy-note"><span className="lock-icon">⌁</span> Runs privately on this device</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="mobile-brand"><div className="brand-mark"><Icon name="spark" size={18} /></div><strong>Local AI Studio</strong></div><div className="connection-pill"><span className={`status-dot ${isOnline ? "online" : "offline"}`} />{statusLoading ? "Checking runtime" : isOnline ? "Local runtime ready" : "Runtime unavailable"}</div></header>
        {selectedTask === "ocr" ? <OCRWorkspace /> : selectedTask === "embeddings" ? <RAGWorkspace models={models} /> : <section className="workspace">
          <div className="workspace-heading"><div><p className="eyebrow">TASK-AWARE LOCAL AI</p><h1>{activeTask?.display_name || "Local model workspace"}</h1><p className="subtitle">Choose a task, review the compatible local models, and run privately on this machine.</p></div><button className="clear-button" onClick={() => setMessages([])} disabled={!messages.length || isLoading}><Icon name="trash" size={15} /> Clear chat</button></div>

          <div className="task-picker"><div className="task-picker-label">TASK</div><div className="task-tabs">{tasks.map((task) => <button className={`task-tab ${task.id === selectedTask ? "selected" : ""}`} key={task.id} onClick={() => changeTask(task.id)}><Icon name={taskIcons[task.id] || "spark"} size={16} /><span>{task.display_name}</span>{task.id === selectedTask && <Icon name="check" size={14} />}</button>)}</div></div>

          <div className="model-panel">
            <div className="model-toolbar"><div className="model-control"><label htmlFor="model-select">MODEL FOR THIS TASK</label><div className="select-wrap"><Icon name="cpu" size={16} /><select id="model-select" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} disabled={modelsLoading || !taskModels.length}><option value="">{modelsLoading ? "Loading model registry..." : taskModels.length ? "Select a compatible model" : "No catalog models for this task"}</option>{taskModels.map((model) => <option key={model.identifier} value={model.identifier} disabled={!model.installed}>{modelLabel(model)} · {model.identifier} {!model.installed ? " · Unavailable" : ""}</option>)}</select></div></div><button className="refresh-button" onClick={refreshModels} disabled={modelsLoading}><Icon name="refresh" size={15} /> Refresh registry</button></div>
            {activeModel && <div className="model-summary"><div className="model-summary-main"><div className="model-avatar"><Icon name="cpu" size={18} /></div><div><strong>{modelLabel(activeModel)}</strong><span>{activeModel.identifier} · {activeModel.provider}</span></div></div><div className="model-badges"><span className={`badge ${activeModel.installed ? "installed" : "missing"}`}>{activeModel.installed ? "Installed" : "Unavailable"}</span><span className="badge neutral">{activeModel.license_info?.identifier || "License unverified"}</span><span className="badge neutral">{activeModel.runtime_status === "installed_not_loaded" ? "Installed · not loaded" : activeModel.runtime_status}</span></div></div>}
            {preferredMissing && activeTask && <div className="notice warning"><span>!</span><div><strong>Preferred model unavailable:</strong> {activeTask.preferred_model} is not installed. Choose an installed alternative above; the app will not silently switch models.</div></div>}
            {activeTask && !activeTask.pipeline_available && <div className="notice info"><span>i</span><div><strong>Planned workflow:</strong> {activeTask.pipeline_note} Models are cataloged now, but this task will not send files or prompts to the chat endpoint.</div></div>}
          </div>

          {(modelsError || statusError) && <div className="alert error-alert"><span>!</span><div>{modelsError || statusError}<button onClick={() => { refreshModels(); refreshHealth(); }}>Try again</button></div></div>}

          <div className={`conversation ${messages.length ? "has-messages" : ""}`}>
            {!messages.length && <div className="empty-state"><div className="empty-icon"><Icon name={taskIcons[selectedTask] || "spark"} size={27} /></div><h2>{activeTask?.pipeline_available ? `Ready for ${activeTask.display_name.toLowerCase()}` : `${activeTask?.display_name || "Task"} is coming next`}</h2><p>{activeTask?.pipeline_available ? "Select an installed model and start a private conversation." : activeTask?.pipeline_note}</p>{activeTask?.pipeline_available && <div className="suggestions"><button onClick={() => setDraft(selectedTask === "coding" ? "Explain this code and suggest improvements:\n\n" : "Explain how local inference works in simple terms.")}>{selectedTask === "coding" ? "Review code" : "Explain local inference"}</button><button onClick={() => setDraft("Help me brainstorm a project idea.")}>Brainstorm an idea</button></div>}</div>}
            {messages.map((message, index) => <div className={`message-row ${message.role}`} key={`${message.role}-${index}`}><div className="avatar">{message.role === "user" ? "You" : <Icon name="spark" size={15} />}</div><div className="message-body"><div className="message-label">{message.role === "user" ? "You" : message.model || "Local AI"}</div><div className="message-content">{message.content}</div></div></div>)}
            {isLoading && <div className="message-row assistant"><div className="avatar"><Icon name="spark" size={15} /></div><div className="message-body"><div className="message-label">{selectedModel}</div><div className="typing"><span /><span /><span /></div></div></div>}
          </div>

          {chatError && <div className="alert error-alert chat-alert"><span>!</span><div>{chatError}</div></div>}
          {activeTask?.pipeline_available && <form className="composer" onSubmit={handleSubmit}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder={selectedModel ? `Message ${modelLabel(activeModel || { identifier: selectedModel })}...` : "Select an installed model to begin..."} disabled={!taskCanChat || isLoading} rows="1" aria-label="Message" /><div className="composer-footer"><span>Enter to send · Shift + Enter for a new line</span><button className="send-button" type="submit" disabled={!draft.trim() || !taskCanChat || isLoading}><span>{isLoading ? "Thinking" : "Send"}</span><Icon name="send" size={15} /></button></div></form>}
          <p className="disclaimer">Local runtime only · AI responses can be inaccurate. Verify important information.</p>
        </section>}
      </main>
    </div>
  );
}

export default App;
