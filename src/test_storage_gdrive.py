# test_storage_gdrive.py
"""
Unit tests for ModelFloat MVP Google Drive storage adapter.
Uses mocking to avoid real API/network calls.
"""

import sys
import types
import pytest

# Patch sys.modules to provide dummy Google API modules for import
googleapiclient_mod = types.ModuleType("googleapiclient")
googleapiclient_mod.discovery = types.ModuleType("discovery")
googleapiclient_mod.http = types.ModuleType("http")
sys.modules["googleapiclient"] = googleapiclient_mod
sys.modules["googleapiclient.discovery"] = googleapiclient_mod.discovery
sys.modules["googleapiclient.http"] = googleapiclient_mod.http

google_mod = types.ModuleType("google")
google_mod.oauth2 = types.ModuleType("oauth2")
google_mod.oauth2.service_account = types.ModuleType("service_account")
sys.modules["google"] = google_mod
sys.modules["google.oauth2"] = google_mod.oauth2
sys.modules["google.oauth2.service_account"] = google_mod.oauth2.service_account

# Dummy classes for patching
class DummyServiceAccountCreds:
    @staticmethod
    def from_service_account_file(path, scopes=None):
        return "dummy_creds"

class DummyDriveService:
    def __init__(self):
        self.files_resource = self
    def files(self):
        return self
    def create(self, body, media_body, fields):
        class DummyExec:
            def execute(self_inner):
                return {"id": "file123"}
        return DummyExec()
    def get_media(self, fileId):
        class DummyRequest:
            def __init__(self):
                self.fileId = fileId
        return DummyRequest()

class DummyMediaFileUpload:
    def __init__(self, path, resumable=True):
        self.path = path

class DummyMediaIoBaseDownload:
    def __init__(self, f, request):
        self.f = f
        self.request = request
        self.called = False
    def next_chunk(self):
        if not self.called:
            self.called = True
            return (None, True)
        return (None, True)

# Patch the imported modules/classes
googleapiclient_mod.discovery.build = lambda *a, **kw: DummyDriveService()
googleapiclient_mod.http.MediaFileUpload = DummyMediaFileUpload
google_mod.oauth2.service_account.Credentials = DummyServiceAccountCreds
sys.modules["googleapiclient.http"].MediaIoBaseDownload = DummyMediaIoBaseDownload

from storage_gdrive import GoogleDriveStorageAdapter

def test_upload_fragment(monkeypatch):
    # Patch build to DummyDriveService for this test only
    monkeypatch.setattr(googleapiclient_mod.discovery, "build", lambda *a, **kw: DummyDriveService())
    adapter = GoogleDriveStorageAdapter("dummy.json")
    file_id = adapter.upload_fragment("fragment.bin")
    assert file_id == "file123"

def test_upload_fragment_error(monkeypatch):
    class ErrorDriveService(DummyDriveService):
        def create(self, body, media_body, fields):
            class DummyExec:
                def execute(self_inner):
                    raise Exception("Upload failed")
            return DummyExec()
    adapter = GoogleDriveStorageAdapter("dummy.json")
    adapter.service = ErrorDriveService()
    file_id = adapter.upload_fragment("fragment.bin")
    assert file_id is None

def test_download_fragment(monkeypatch, tmp_path):
    # Patch build to DummyDriveService and patch MediaIoBaseDownload in sys.modules
    monkeypatch.setattr(googleapiclient_mod.discovery, "build", lambda *a, **kw: DummyDriveService())
    sys.modules["googleapiclient.http"].MediaIoBaseDownload = DummyMediaIoBaseDownload
    adapter = GoogleDriveStorageAdapter("dummy.json")
    file_id = "file123"
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment(file_id, str(dest))
    assert ok
    assert dest.exists()

def test_download_fragment_error(monkeypatch, tmp_path):
    class ErrorDriveService(DummyDriveService):
        def get_media(self, fileId):
            raise Exception("Download failed")
    sys.modules["googleapiclient.http"].MediaIoBaseDownload = DummyMediaIoBaseDownload
    adapter = GoogleDriveStorageAdapter("dummy.json")
    adapter.service = ErrorDriveService()
    file_id = "file123"
    dest = tmp_path / "out.bin"
    ok = adapter.download_fragment(file_id, str(dest))
    assert not ok
