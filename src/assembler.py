# assembler.py
"""
ModelFloat MVP — Fragment Assembler

Retrieves fragments from storage providers and reassembles the original file.
For the MVP, expects a list of fragment metadata dicts with provider and location.
"""

from typing import List, Dict, Callable

class FragmentAssembler:
    def __init__(self, providers: Dict[str, Callable]):
        """
        providers: dict mapping provider name to adapter instance
        Example: {"mega": mega_adapter, "gdrive": gdrive_adapter, "ipfs": ipfs_adapter}
        """
        self.providers = providers

    def retrieve_and_assemble(self, fragments: List[Dict], out_path: str) -> bool:
        """
        Downloads all fragments and reassembles them into out_path.
        fragments: list of dicts with keys: fragment_id, index, size_bytes, checksum, storage (list of locations)
        Returns True on success, False on failure.
        """
        # Sort fragments by index
        fragments_sorted = sorted(fragments, key=lambda f: f["index"])
        try:
            with open(out_path, "wb") as out_f:
                for frag in fragments_sorted:
                    # Try each storage location until one succeeds
                    data = None
                    for loc in frag["storage"]:
                        provider = loc["provider"]
                        location = loc["location"]
                        adapter = self.providers.get(provider)
                        if not adapter:
                            continue
                        tmp_path = f"tmp_{frag['fragment_id']}"
                        ok = adapter.download_fragment(location, tmp_path)
                        if ok:
                            with open(tmp_path, "rb") as f:
                                data = f.read()
                            break
                    if data is None:
                        print(f"Failed to retrieve fragment {frag['fragment_id']}")
                        return False
                    out_f.write(data)
            return True
        except Exception as e:
            print(f"Assembly failed: {e}")
            return False
