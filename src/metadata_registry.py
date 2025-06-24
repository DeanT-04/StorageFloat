# metadata_registry.py
"""
ModelFloat MVP — Metadata Registry

Implements a minimal metadata registry using a JSON file.
Tracks models, fragments, and storage locations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

METADATA_FILE = os.path.join(os.path.dirname(__file__), "metadata.json")


def _load_metadata() -> Dict:
    if not os.path.exists(METADATA_FILE):
        return {"models": [], "fragments": []}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def _save_metadata(data: Dict):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_model(model_id: str, name: str, description: str, framework: str, size_bytes: int):
    data = _load_metadata()
    model = {
        "model_id": model_id,
        "name": name,
        "description": description,
        "framework": framework,
        "size_bytes": size_bytes,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    data["models"].append(model)
    _save_metadata(data)
    return model


def list_models() -> List[Dict]:
    data = _load_metadata()
    return data.get("models", [])


def add_fragment(fragment_id: str, model_id: str, index: int, size_bytes: int, checksum: str, storage: List[Dict]):
    data = _load_metadata()
    fragment = {
        "fragment_id": fragment_id,
        "model_id": model_id,
        "index": index,
        "size_bytes": size_bytes,
        "checksum": checksum,
        "storage": storage
    }
    data["fragments"].append(fragment)
    _save_metadata(data)
    return fragment


def list_fragments(model_id: Optional[str] = None) -> List[Dict]:
    data = _load_metadata()
    if model_id:
        return [f for f in data.get("fragments", []) if f["model_id"] == model_id]
    return data.get("fragments", [])


def get_model(model_id: str) -> Optional[Dict]:
    data = _load_metadata()
    for m in data.get("models", []):
        if m["model_id"] == model_id:
            return m
    return None


def get_fragment(fragment_id: str) -> Optional[Dict]:
    data = _load_metadata()
    for f in data.get("fragments", []):
        if f["fragment_id"] == fragment_id:
            return f
    return None
