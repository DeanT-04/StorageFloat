# fragmenter.py
"""
ModelFloat MVP — Model Fragmenter

Splits a model file into fixed-size fragments for distributed storage.
For the MVP, works with any file (simulate with small files).
"""

import os
import hashlib
from typing import List, Dict

FRAGMENT_SIZE = 1024 * 1024  # 1 MB for MVP

def fragment_file(filepath: str) -> List[Dict]:
    """
    Splits the file at `filepath` into fragments of FRAGMENT_SIZE bytes.
    Returns a list of fragment metadata dicts:
      - fragment_id: SHA256 hash of fragment data
      - index: fragment order
      - size_bytes: size of fragment
      - checksum: SHA256 hash
      - data: bytes (for storage/upload)
    """
    fragments = []
    with open(filepath, "rb") as f:
        index = 0
        while True:
            chunk = f.read(FRAGMENT_SIZE)
            if not chunk:
                break
            checksum = hashlib.sha256(chunk).hexdigest()
            fragment_id = checksum  # Use checksum as ID for MVP
            fragments.append({
                "fragment_id": fragment_id,
                "index": index,
                "size_bytes": len(chunk),
                "checksum": checksum,
                "data": chunk
            })
            index += 1
    return fragments

def save_fragments(fragments: List[Dict], out_dir: str, base_name: str):
    """
    Saves each fragment's data as a file in `out_dir` with names like base_name.part0, base_name.part1, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    for frag in fragments:
        fname = f"{base_name}.part{frag['index']}"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(frag["data"])
