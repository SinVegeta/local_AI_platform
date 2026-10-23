"""Versioned local benchmark corpus manifest."""

import json
from pathlib import Path


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "1.0" or not isinstance(data.get("documents"), list):
        raise ValueError("Benchmark manifest must use version 1.0 and contain documents")
    root = path.parent.resolve()
    for document in data["documents"]:
        for required in ("document_id", "path"):
            if not document.get(required):
                raise ValueError(f"Benchmark document is missing {required}")
        source = (root / document["path"]).resolve()
        if root not in source.parents and source != root:
            raise ValueError(f"Benchmark path escapes manifest directory: {document['path']}")
        document["resolved_path"] = str(source)
        for page in document.get("pages", []):
            if "page_number" not in page:
                raise ValueError(f"Benchmark page is missing page_number for {document['document_id']}")
            reference = page.get("reference_path") or page.get("ground_truth_path")
            if reference:
                reference_path = (root / reference).resolve()
                if root not in reference_path.parents and reference_path != root:
                    raise ValueError(f"Ground-truth path escapes manifest directory: {reference}")
                page["resolved_reference_path"] = str(reference_path)
        if document.get("ground_truth_path"):
            reference_path = (root / document["ground_truth_path"]).resolve()
            if root not in reference_path.parents and reference_path != root:
                raise ValueError(f"Ground-truth path escapes manifest directory: {document['ground_truth_path']}")
            document["resolved_ground_truth_path"] = str(reference_path)
    return data


def validate_manifest(path: Path) -> list[str]:
    data = load_manifest(path)
    errors = []
    for document in data["documents"]:
        if not Path(document["resolved_path"]).is_file():
            errors.append(f"missing document: {document['path']}")
        if document.get("resolved_ground_truth_path") and not Path(document["resolved_ground_truth_path"]).is_file():
            errors.append(f"missing ground truth: {document['ground_truth_path']}")
        for page in document.get("pages", []):
            reference = page.get("resolved_reference_path")
            if reference and not Path(reference).is_file():
                errors.append(f"missing ground truth: {page.get('reference_path') or page.get('ground_truth_path')}")
    return errors


def write_manifest(path: Path, documents: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "1.0", "documents": documents}, indent=2, ensure_ascii=False), encoding="utf-8")

