# storage_mega.py
"""
ModelFloat MVP — MEGA Storage Adapter

Handles upload and download of fragments to/from MEGA cloud storage.
Requires the 'mega.py' package and MEGA account credentials.
"""

from mega import Mega
import os
from typing import Optional

class MegaStorageAdapter:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password
        self.mega = Mega()
        self.m = None

    def connect(self):
        """Login to MEGA account."""
        if not self.m:
            self.m = self.mega.login(self.email, self.password)

    def upload_fragment(self, fragment_path: str) -> Optional[str]:
        """
        Uploads a fragment file to MEGA.
        Returns the public link to the uploaded file, or None on failure.
        """
        self.connect()
        try:
            file = self.m.upload(fragment_path)
            link = self.m.get_upload_link(file)
            return link
        except Exception as e:
            print(f"MEGA upload failed: {e}")
            return None

    def download_fragment(self, file_link: str, dest_path: str) -> bool:
        """
        Downloads a fragment from MEGA using its public link.
        Saves to dest_path. Returns True on success, False on failure.
        """
        self.connect()
        try:
            self.m.download_url(file_link, dest_path)
            return True
        except Exception as e:
            print(f"MEGA download failed: {e}")
            return False
