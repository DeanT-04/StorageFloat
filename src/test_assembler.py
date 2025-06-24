# test_assembler.py
"""
Unit tests for ModelFloat MVP assembler.py
"""

import os
from assembler import FragmentAssembler

class DummyAdapter:
    def __init__(self, data_map):
        self.data_map = data_map
        self.downloaded = []
    def download_fragment(self, location, dest_path):
        if location in self.data_map:
            with open(dest_path, "wb") as f:
                f.write(self.data_map[location])
            self.downloaded.append(location)
            return True
        return False

def test_retrieve_and_assemble_success(tmp_path):
    # Prepare dummy data
    frag1 = b"A" * 10
    frag2 = b"B" * 5
    adapter = DummyAdapter({
        "loc1": frag1,
        "loc2": frag2
    })
    fragments = [
        {
            "fragment_id": "f1",
            "index": 0,
            "size_bytes": 10,
            "checksum": "",
            "storage": [{"provider": "dummy", "location": "loc1"}]
        },
        {
            "fragment_id": "f2",
            "index": 1,
            "size_bytes": 5,
            "checksum": "",
            "storage": [{"provider": "dummy", "location": "loc2"}]
        }
    ]
    assembler = FragmentAssembler({"dummy": adapter})
    out_path = tmp_path / "assembled.bin"
    ok = assembler.retrieve_and_assemble(fragments, str(out_path))
    assert ok
    assert out_path.read_bytes() == frag1 + frag2

def test_retrieve_and_assemble_missing_fragment(tmp_path):
    adapter = DummyAdapter({"loc1": b"A"})
    fragments = [
        {
            "fragment_id": "f1",
            "index": 0,
            "size_bytes": 1,
            "checksum": "",
            "storage": [{"provider": "dummy", "location": "loc1"}]
        },
        {
            "fragment_id": "f2",
            "index": 1,
            "size_bytes": 1,
            "checksum": "",
            "storage": [{"provider": "dummy", "location": "missing"}]
        }
    ]
    assembler = FragmentAssembler({"dummy": adapter})
    out_path = tmp_path / "fail.bin"
    ok = assembler.retrieve_and_assemble(fragments, str(out_path))
    assert not ok
