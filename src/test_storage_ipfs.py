# test_storage_ipfs.py
"""
Unit tests for ModelFloat MVP IPFS storage adapter.
Uses mocking to avoid real network calls.
"""

import sys
import types
import pytest

# Patch sys.modules to provide a dummy ipfshttpclient module for import
ipfs_mod = types.ModuleType("ipfshttpclient")
sys.modules["ipfshttpclient"] = ipfs_mod

class DummyIPFSClient:
    def __init__(self):
        self.uploaded = {}
    def add(self, path):
        self.uploaded[path] = True
        return {"Hash": "QmFakeHash"}
    def get(self, cid, target):
        if cid == "fail":
            raise Exception("Download failed")
        with open(target, "wb") as f:
            f.write(b"data")

def dummy_connect(api_url):
    return DummyIPFSClient()

ipfs_mod.connect = dummy_connect

from storage_ipfs import IPFSStorageAdapter

def test_upload_fragment(monkeypatch):
    adapter = IPFSStorageAdapter()
    cid = adapter.upload_fragment("fragment.bin")
    assert cid == "QmFakeHash"

def test_upload_fragment_error(monkeypatch):
    class ErrorIPFSClient(DummyIPFSClient):
        def add(self, path):
            raise Exception("Upload failed")
    monkeypatch.setattr(ipfs_mod, "connect", lambda api_url: ErrorIPFSClient())
    adapter = IPFSStorageAdapter()
    cid = adapter.upload_fragment("fragment.bin")
    assert cid is None

def test_download_fragment(monkeypatch, tmp_path):
    adapter = IPFSStorageAdapter()
    cid = "QmFakeHash"
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment(cid, str(dest))
    assert ok
    assert dest.read_bytes() == b"data"

def test_download_fragment_error(monkeypatch, tmp_path):
    class ErrorIPFSClient(DummyIPFSClient):
        def get(self, cid, target):
            raise Exception("Download failed")
    monkeypatch.setattr(ipfs_mod, "connect", lambda api_url: ErrorIPFSClient())
    adapter = IPFSStorageAdapter()
    cid = "fail"
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment(cid, str(dest))
    assert not ok
