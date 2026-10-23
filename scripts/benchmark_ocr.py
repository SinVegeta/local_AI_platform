"""Run the reproducible local OCR benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from app.benchmark.manifest import load_manifest, validate_manifest, write_manifest
from app.benchmark.metrics import compare_text
from app.core.config import get_settings
from app.providers.glm_ocr import GLMOCRProvider
from app.providers.ocr_base import OCRProviderError
from app.providers.tesseract import TesseractProvider
from app.schemas.documents import OCRConfig
from app.services.pdf_extraction import PDFExtractor, PDFProcessingError


def _gpu_memory() -> dict | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        used, free = [int(part.strip()) for part in result.stdout.splitlines()[0].split(",")[:2]]
        return {"used_mb": used, "free_mb": free}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _image_fixture(path: Path, text: str) -> None:
    image = Image.new("RGB", (1600, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 150), text, fill="black", font=ImageFont.load_default(size=48))
    image.save(path)


def generate_synthetic(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "scanned-page.png"
    _image_fixture(image_path, "Synthetic OCR benchmark. Page 2: local extraction.")
    jpeg_path = root / "scanned-page.jpg"
    Image.open(image_path).save(jpeg_path, quality=95)
    import pymupdf

    digital = root / "digital.pdf"
    doc = pymupdf.open()
    for text in ("Synthetic digital PDF. Page 1: selectable text.", "Synthetic digital PDF. Page 2: second page."):
        page = doc.new_page()
        page.insert_text((72, 100), text, fontsize=18)
    doc.save(digital)
    doc.close()

    scanned = root / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image_path))
    doc.save(scanned)
    doc.close()

    mixed = root / "mixed.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Synthetic mixed PDF. Page 1: native text.", fontsize=18)
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image_path))
    doc.save(mixed)
    doc.close()
    manifest = root / "manifest.json"
    write_manifest(manifest, [
        {"document_id": "synthetic-digital", "path": digital.name, "document_type": "digital_pdf", "language": "eng", "pages": [{"page_number": 1, "expected_text": "Synthetic digital PDF. Page 1: selectable text."}, {"page_number": 2, "expected_text": "Synthetic digital PDF. Page 2: second page."}]},
        {"document_id": "synthetic-scanned", "path": scanned.name, "document_type": "scanned_pdf", "language": "eng", "pages": [{"page_number": 1, "expected_text": "Synthetic OCR benchmark. Page 2: local extraction."}]},
        {"document_id": "synthetic-mixed", "path": mixed.name, "document_type": "mixed_pdf", "language": "eng", "pages": [{"page_number": 1, "expected_text": "Synthetic mixed PDF. Page 1: native text."}, {"page_number": 2, "expected_text": "Synthetic OCR benchmark. Page 2: local extraction."}]},
        {"document_id": "synthetic-image", "path": image_path.name, "document_type": "image", "language": "eng", "pages": [{"page_number": 1, "expected_text": "Synthetic OCR benchmark. Page 2: local extraction."}]},
        {"document_id": "synthetic-jpeg", "path": jpeg_path.name, "document_type": "image", "language": "eng", "pages": [{"page_number": 1, "expected_text": "Synthetic OCR benchmark. Page 2: local extraction."}]},
    ])
    return manifest


def _pages(source: Path, extractor: PDFExtractor | None):
    if source.suffix.lower() != ".pdf":
        return [(1, None, None)]
    assert extractor
    with extractor.open(source) as document:
        return [(index + 1, extractor.usable_text(page), extractor.render_page(page, 150)) for index, page in enumerate(document)]


def _expected_text(document: dict, page: dict, root: Path) -> tuple[str | None, str | None]:
    if "expected_text" in page:
        return page["expected_text"], "inline"
    reference = page.get("resolved_reference_path")
    if reference:
        return Path(reference).read_text(encoding="utf-8"), page.get("reference_path") or page.get("ground_truth_path")
    if document.get("resolved_ground_truth_path") and page["page_number"] == 1:
        return Path(document["resolved_ground_truth_path"]).read_text(encoding="utf-8"), document.get("ground_truth_path")
    return None, None


def run(manifest_path: Path, engines: list[str], output_dir: Path, selected_ids: set[str] | None = None) -> Path:
    manifest = load_manifest(manifest_path)
    settings = get_settings()
    config = OCRConfig(primary_engine="tesseract", language="en", enable_fallback=False, render_dpi=150)
    extractor = PDFExtractor()
    providers = {"tesseract": TesseractProvider(settings), "glm-ocr": GLMOCRProvider(settings)}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    selected_documents = [document for document in manifest["documents"] if not selected_ids or document["document_id"] in selected_ids]
    for document in selected_documents:
        source = Path(document["resolved_path"])
        expected = {page["page_number"]: page for page in document.get("pages", [])}
        try:
            page_data = _pages(source, extractor)
        except (PDFProcessingError, OSError) as exc:
            results.append({"document_id": document["document_id"], "path": document["path"], "status": "failed", "error": str(exc), "pages": []})
            continue
        for engine in engines:
            engine_pages = []
            doc_started = time.perf_counter()
            for page_number, native_text, rendered in page_data:
                truth = expected.get(page_number, {})
                started = time.perf_counter()
                result = {"page_number": page_number, "engine": engine, "status": "not_applicable"}
                try:
                    prediction = ""
                    method = None
                    version = None
                    applicable = True
                    if engine == "native":
                        if source.suffix.lower() != ".pdf" or not native_text:
                            result["reason"] = "Native extraction applies only to PDF pages with selectable text."
                            applicable = False
                        else:
                            prediction, method, version = native_text, "native_pdf_text", extractor.version
                    else:
                        with tempfile.TemporaryDirectory(prefix="ocr-benchmark-") as work:
                            image_path = Path(work) / f"page-{page_number}.png"
                            image_path.write_bytes(rendered if rendered is not None else source.read_bytes())
                            page_result = providers[engine].extract(image_path, page_number, config)
                            prediction, method, version = page_result.text, page_result.extraction_method, page_result.engine_version
                            result["warnings"] = page_result.warnings
                    if applicable:
                        result.update({"status": "completed", "text": prediction, "extraction_method": method, "engine_version": version})
                    expected_text, ground_truth_source = _expected_text(document, truth, Path(manifest_path).parent)
                    result["ground_truth_available"] = expected_text is not None
                    if expected_text is not None and applicable:
                        result["ground_truth_source"] = ground_truth_source
                        result["metrics"] = compare_text(expected_text, prediction)
                    if applicable:
                        result["manual_review"] = not bool(prediction.strip())
                except (OCRProviderError, KeyError, OSError, PDFProcessingError) as exc:
                    result.update({"status": "failed", "error": str(exc), "manual_review": True})
                result["duration_ms"] = int((time.perf_counter() - started) * 1000)
                engine_pages.append(result)
            completed = sum(page["status"] == "completed" for page in engine_pages)
            duration_ms = int((time.perf_counter() - doc_started) * 1000)
            results.append({"document_id": document["document_id"], "path": document["path"], "document_type": document.get("document_type"), "engine": engine, "engine_version": extractor.version if engine == "native" else ("glm-ocr:latest" if engine == "glm-ocr" else "Tesseract"), "status": "completed" if completed == len(engine_pages) else "partially_completed", "duration_ms": duration_ms, "pages_per_minute": round(completed / (duration_ms / 60000), 3) if duration_ms else None, "failed_pages": sum(page["status"] == "failed" for page in engine_pages), "manual_review_pages": sum(page.get("manual_review", False) for page in engine_pages), "pages": engine_pages})
    comparable_pages = [page for item in results for page in item.get("pages", []) if page.get("metrics")]
    by_engine = {engine: {"attempted_pages": sum(1 for item in results for page in item.get("pages", []) if page.get("engine") == engine and page.get("status") != "not_applicable"), "failed_pages": sum(1 for item in results for page in item.get("pages", []) if page.get("engine") == engine and page.get("status") == "failed")} for engine in engines}
    for values in by_engine.values():
        values["failure_rate"] = values["failed_pages"] / values["attempted_pages"] if values["attempted_pages"] else None
    payload = {"version": "1.0", "run_id": output_dir.name, "benchmark_started_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "python": sys.version, "engines": engines, "runtime": {"glm_model": "glm-ocr:latest", "glm_load_state": "not queried; Ollama load state is not exposed by this benchmark", "execution": "sequential", "hardware": "NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM"}, "gpu_memory_before": _gpu_memory(), "results": results, "summary": {"available_documents": len(manifest["documents"]), "selected_documents": len(selected_documents), "documents": len({item["document_id"] for item in results}), "engine_runs": len(results), "comparable_pages": len(comparable_pages), "ground_truth_pages": sum(1 for item in results for page in item.get("pages", []) if page.get("ground_truth_available")), "failed_pages": sum(item.get("failed_pages", 0) for item in results), "manual_review_pages": sum(item.get("manual_review_pages", 0) for item in results), "by_engine": by_engine, "fallback_frequency": None, "fallback_frequency_note": "Direct benchmark runs do not invoke production fallback.", "process_ram_peak_mb": None}, "notes": ["All engine runs are sequential and local.", "GLM-OCR is run directly; production fallback frequency is not inferred from direct comparison runs.", "Process RAM peak is unavailable without an additional package and is reported as unavailable."]}
    payload["gpu_memory_after"] = _gpu_memory()
    output = output_dir / "results.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(payload, output_dir / "report.md")
    return output


def _write_report(payload: dict, path: Path) -> None:
    lines = ["# Local OCR Benchmark", "", f"Started: `{payload['benchmark_started_at']}`", f"Engines: `{', '.join(payload['engines'])}`", "", "| Document | Engine | Pages | Exact | CER | WER |", "|---|---|---:|---:|---:|---:|"]
    for item in payload["results"]:
        pages = [page for page in item.get("pages", []) if page.get("metrics")]
        exact = sum(page["metrics"]["exact_match"] for page in pages) / len(pages) if pages else None
        cer = sum(page["metrics"]["cer"] for page in pages if page["metrics"]["cer"] is not None) / len(pages) if pages else None
        wer = sum(page["metrics"]["wer"] for page in pages if page["metrics"]["wer"] is not None) / len(pages) if pages else None
        lines.append(f"| {item['document_id']} | {item.get('engine', 'n/a')} | {len(item.get('pages', []))} | {exact if exact is not None else 'n/a'} | {cer if cer is not None else 'n/a'} | {wer if wer is not None else 'n/a'} |")
    summary = payload["summary"]
    coverage = f"{summary.get('ground_truth_pages', 0)}/{summary.get('selected_documents', 0)} selected documents/pages association count"
    lines += ["", "## Operational summary", "", f"- Available documents: `{summary.get('available_documents', 'n/a')}`; selected: `{summary.get('selected_documents', 'n/a')}`; comparable pages: `{summary['comparable_pages']}`.", f"- Ground-truth page associations: `{summary.get('ground_truth_pages', 0)}`; failed pages: `{summary['failed_pages']}`; manual-review flags: `{summary['manual_review_pages']}`.", "- Per-engine failure rates are stored in `results.json` under `summary.by_engine`.", "- Fallback frequency: `n/a` because direct benchmark runs intentionally do not invoke production fallback.", f"- GPU memory before: `{payload.get('gpu_memory_before') or 'unavailable'}`; after: `{payload.get('gpu_memory_after') or 'unavailable'}`.", "- Process RAM peak: `unavailable` without adding a dependency.", "", "## Measurement notes", "", "- Native PyMuPDF is measured as digital extraction, not OCR.", "- Direct engine comparison does not automatically invoke production fallback; fallback frequency is not inferred.", "- Manual review is flagged for failed or empty predictions only.", "- GPU memory is recorded only when `nvidia-smi` is available."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--generate-synthetic", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocr-benchmarks/runs/latest"))
    parser.add_argument("--engines", default="native,tesseract,glm-ocr")
    parser.add_argument("--list", action="store_true", help="List manifest documents and availability")
    parser.add_argument("--validate", "--dry-run", action="store_true", help="Validate manifest and local references without inference")
    parser.add_argument("--report-from", type=Path, help="Regenerate Markdown from saved results.json")
    parser.add_argument("--ids", help="Comma-separated document IDs to run")
    parser.add_argument("--quick", action="store_true", help="Run only the first selected document")
    args = parser.parse_args()
    if args.report_from:
        payload = json.loads(args.report_from.read_text(encoding="utf-8"))
        report_path = args.report_from.with_name("report.md")
        _write_report(payload, report_path)
        print(report_path)
        return
    manifest = args.manifest or (generate_synthetic(args.generate_synthetic) if args.generate_synthetic else None)
    if not manifest:
        parser.error("Provide --manifest or --generate-synthetic (or --report-from)")
    data = load_manifest(manifest)
    if args.list:
        for document in data["documents"]:
            source = Path(document["resolved_path"])
            print(f"{document['document_id']}\t{document.get('document_type', 'unknown')}\t{'available' if source.is_file() else 'missing'}\t{document['path']}")
        return
    errors = validate_manifest(manifest)
    if args.validate:
        print("Manifest valid" if not errors else "Manifest has validation issues:")
        for error in errors:
            print(f"- {error}")
        return
    if args.generate_synthetic:
        print(f"Generated benchmark manifest: {manifest}")
    selected_ids = {item.strip() for item in args.ids.split(",")} if args.ids else None
    if args.quick:
        selected_ids = {next((item["document_id"] for item in data["documents"] if not selected_ids or item["document_id"] in selected_ids), data["documents"][0]["document_id"])}
    if errors:
        print("Warning: missing local files will be recorded as failed/not available:")
        for error in errors:
            print(f"- {error}")
    print(run(manifest, [engine.strip() for engine in args.engines.split(",") if engine.strip()], args.output_dir, selected_ids))


if __name__ == "__main__":
    main()
