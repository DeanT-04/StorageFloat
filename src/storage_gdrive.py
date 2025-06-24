# storage_gdrive.py
"""
ModelFloat MVP — Google Drive Storage Adapter

Handles upload and download of fragments to/from Google Drive.
Requires 'google-api-python-client', 'google-auth-httplib2', and 'google-auth-oauthlib'.
"""

import os
from typing import Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials

class GoogleDriveStorageAdapter:
    def __init__(self, credentials_json: str):
        """
        credentials_json: Path to Google service account JSON credentials.
        """
        self.credentials_json = credentials_json
        self.creds = Credentials.from_service_account_file(credentials_json, scopes=["https://www.googleapis.com/auth/drive"])
        self.service = build("drive", "v3", credentials=self.creds)

    def upload_fragment(self, fragment_path: str, folder_id: Optional[str] = None) -> Optional[str]:
        """
        Uploads a fragment file to Google Drive.
        Returns the file ID or None on failure.
        """
        file_metadata = {"name": os.path.basename(fragment_path)}
        if folder_id:
            file_metadata["parents"] = [folder_id]
        media = MediaFileUpload(fragment_path, resumable=True)
        try:
            file = self.service.files().create(body=file_metadata, media_body=media, fields="id").execute()
            return file.get("id")
        except Exception as e:
            print(f"Google Drive upload failed: {e}")
            return None

    def download_fragment(self, file_id: str, dest_path: str) -> bool:
        """
        Downloads a fragment from Google Drive by file ID.
        Saves to dest_path. Returns True on success, False on failure.
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            with open(dest_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            return True
        except Exception as e:
            print(f"Google Drive download failed: {e}")
            return False

# Note: For MVP testing, use mock objects or patch the Google API in unit tests.
