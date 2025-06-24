# test_storage_mega.py
"""
Unit tests for ModelFloat MVP MEGA storage adapter.
Uses mocking to avoid real network calls.
"""

import pytest
import sys
import types

# Patch sys.modules to provide a dummy 'mega' module for import
mega_mod = types.ModuleType("mega")
class DummyMega:
    def __init__(self):
        self.uploaded = {}
    def login(self, email, password):
        return self
    def upload(self, path):
        self.uploaded[path] = True
        return {"name": path}
    def get_upload_link(self, file):
        return f"https://mega.nz/fake/{file['name']}"
    def download_url(self, link, dest):
        if "fail" in link:
            raise Exception("Download failed")
        with open(dest, "wb") as f:
            f.write(b"data")
mega_mod.Mega = DummyMega
sys.modules["mega"] = mega_mod

from storage_mega import MegaStorageAdapter

# DummyMega is now available as mega.Mega for import in storage_mega.py
def test_upload_fragment(monkeypatch):
    adapter = MegaStorageAdapter("user", "pass")
    monkeypatch.setattr(adapter, "mega", DummyMega())
    adapter.m = None  # Force re-login
    link = adapter.upload_fragment("fragment.bin")
    assert link == "https://mega.nz/fake/fragment.bin"

def test_upload_fragment_error(monkeypatch):
    class ErrorMega(DummyMega):
        def upload(self, path):
            raise Exception("Upload failed")
    adapter = MegaStorageAdapter("user", "pass")
    monkeypatch.setattr(adapter, "mega", ErrorMega())
    adapter.m = None
    link = adapter.upload_fragment("fragment.bin")
    assert link is None

def test_download_fragment(monkeypatch, tmp_path):
    adapter = MegaStorageAdapter("user", "pass")
    monkeypatch.setattr(adapter, "mega", DummyMega())
    adapter.m = None
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment("https://mega.nz/fake/fragment.bin", str(dest))
    assert ok
    assert dest.read_bytes() == b"data"

def test_download_fragment_error(monkeypatch, tmp_path):
    adapter = MegaStorageAdapter("user", "pass")
    monkeypatch.setattr(adapter, "mega", DummyMega())
    adapter.m = None
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment("https://mega.nz/fail/fragment.bin", str(dest))
    assert not ok
