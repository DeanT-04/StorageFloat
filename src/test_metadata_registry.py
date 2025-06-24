# test_metadata_registry.py
"""
Unit tests for ModelFloat MVP metadata_registry.py
"""

import os
import shutil
import uuid
from metadata_registry import (
    add_model, list_models, add_fragment, list_fragments,
    get_model, get_fragment, METADATA_FILE
)

def setup_module(module):
    # Backup and clear metadata file before tests
    if os.path.exists(METADATA_FILE):
        shutil.copy(METADATA_FILE, METADATA_FILE + ".bak")
        os.remove(METADATA_FILE)

def teardown_module(module):
    # Restore metadata file after tests
    if os.path.exists(METADATA_FILE + ".bak"):
        shutil.move(METADATA_FILE + ".bak", METADATA_FILE)
    elif os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)

def test_add_and_get_model():
    model_id = str(uuid.uuid4())
    model = add_model(
        model_id=model_id,
        name="Test Model",
        description="A test model",
        framework="pytorch",
        size_bytes=123456
    )
    assert model["model_id"] == model_id
    fetched = get_model(model_id)
    assert fetched is not None
    assert fetched["name"] == "Test Model"

def test_list_models():
    models = list_models()
    assert isinstance(models, list)

def test_add_and_get_fragment():
    model_id = str(uuid.uuid4())
    add_model(
        model_id=model_id,
        name="Fragmented Model",
        description="Model for fragment test",
        framework="pytorch",
        size_bytes=654321
    )
    fragment_id = str(uuid.uuid4())
    fragment = add_fragment(
        fragment_id=fragment_id,
        model_id=model_id,
        index=0,
        size_bytes=1000,
        checksum="abc123",
        storage=[{"provider": "mega", "location": "url", "uploaded_at": "2025-06-24T13:00:00Z"}]
    )
    assert fragment["fragment_id"] == fragment_id
    fetched = get_fragment(fragment_id)
    assert fetched is not None
    assert fetched["model_id"] == model_id

def test_list_fragments():
    fragments = list_fragments()
    assert isinstance(fragments, list)
