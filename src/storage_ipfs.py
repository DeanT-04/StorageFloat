# storage_ipfs.py
"""
ModelFloat MVP — IPFS Storage Adapter

Handles upload and download of fragments to/from IPFS.
Requires the 'ipfshttpclient' package and a running IPFS node.
"""

import ipfshttpclient
from typing import Optional

class IPFSStorageAdapter:
    def __init__(self, api_url: str = "/ip4/127.0.0.1/tcp/5001"):
        """
        api_url: Multiaddr for the IPFS API (default: local node)
        """
        self.api_url = api_url
        self.client = ipfshttpclient.connect(api_url)

    def upload_fragment(self, fragment_path: str) -> Optional[str]:
        """
        Uploads a fragment file to IPFS.
        Returns the CID (hash) or None on failure.
        """
        try:
            res = self.client.add(fragment_path)
            return res["Hash"]
        except Exception as e:
            print(f"IPFS upload failed: {e}")
            return None

    def download_fragment(self, cid: str, dest_path: str) -> bool:
        """
        Downloads a fragment from IPFS by CID.
        Saves to dest_path. Returns True on success, False on failure.
        """
        try:
            self.client.get(cid, target=dest_path)
            return True
        except Exception as e:
            print(f"IPFS download failed: {e}")
            return False

# Note: For MVP testing, use mock objects or patch ipfshttpclient in unit tests.
