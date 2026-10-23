import json

import pytest

from app.benchmark.manifest import load_manifest, write_manifest
from app.benchmark.metrics import compare_text, normalize_text
from scripts.benchmark_ocr import _write_report


def test_metrics_normalize_and_compare():
    assert normalize_text(" A\n\nB ") == "A B"
    result = compare_text("Hello world", "Hello   world")
    assert result["exact_match"] is True
    assert result["cer"] == 0
    assert result["wer"] == 0


def test_metrics_handle_empty_values_explicitly():
    both = compare_text("", "")
    assert both["comparable"] is False
    assert both["cer"] is None and both["wer"] is None
    missing = compare_text("expected", "")
    assert missing["empty_prediction"] is True
    assert missing["cer"] == 1 and missing["wer"] == 1


def test_manifest_round_trip_and_missing_file(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [{"document_id": "one", "path": "missing.png", "pages": []}])
    data = load_manifest(manifest)
    assert data["documents"][0]["resolved_path"].endswith("missing.png")


def test_manifest_rejects_path_escape(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": "1.0", "documents": [{"document_id": "x", "path": "../outside.png"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest)


def test_utf8_page_reference_is_resolved_and_validated(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not used by this validation test")
    reference = tmp_path / "page-1.txt"
    reference.write_text("नमस्ते local OCR", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, [{"document_id": "unicode", "path": image.name, "pages": [{"page_number": 1, "reference_path": reference.name}]}])
    data = load_manifest(manifest)
    assert data["documents"][0]["pages"][0]["resolved_reference_path"].endswith("page-1.txt")
    assert reference.read_text(encoding="utf-8") in "नमस्ते local OCR"


def test_report_serialization(tmp_path):
    output = tmp_path / "report.md"
    _write_report({"benchmark_started_at": "now", "engines": ["tesseract"], "gpu_memory_before": None, "gpu_memory_after": None, "summary": {"available_documents": 1, "selected_documents": 1, "comparable_pages": 1, "ground_truth_pages": 1, "failed_pages": 0, "manual_review_pages": 0}, "results": [{"document_id": "one", "engine": "tesseract", "pages": [{"metrics": {"exact_match": True, "cer": 0.0, "wer": 0.0}}]}]}, output)
    assert "tesseract" in output.read_text(encoding="utf-8")
