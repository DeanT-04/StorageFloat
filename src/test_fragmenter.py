# test_fragmenter.py
"""
Unit tests for ModelFloat MVP fragmenter.py
"""

import os
import shutil
from fragmenter import fragment_file, save_fragments, FRAGMENT_SIZE

def setup_module(module):
    # Create a test file
    with open("testfile.bin", "wb") as f:
        f.write(b"A" * (FRAGMENT_SIZE + 500))  # 1 full fragment + partial

def teardown_module(module):
    # Remove test file and output dir
    if os.path.exists("testfile.bin"):
        os.remove("testfile.bin")
    if os.path.exists("fragments_out"):
        shutil.rmtree("fragments_out")

def test_fragment_file():
    fragments = fragment_file("testfile.bin")
    assert len(fragments) == 2
    assert fragments[0]["size_bytes"] == FRAGMENT_SIZE
    assert fragments[1]["size_bytes"] == 500
    assert fragments[0]["data"] == b"A" * FRAGMENT_SIZE
    assert fragments[1]["data"] == b"A" * 500

def test_save_fragments():
    fragments = fragment_file("testfile.bin")
    save_fragments(fragments, "fragments_out", "testfile")
    files = os.listdir("fragments_out")
    assert "testfile.part0" in files
    assert "testfile.part1" in files
    with open(os.path.join("fragments_out", "testfile.part0"), "rb") as f:
        data0 = f.read()
    with open(os.path.join("fragments_out", "testfile.part1"), "rb") as f:
        data1 = f.read()
    assert data0 == b"A" * FRAGMENT_SIZE
    assert data1 == b"A" * 500
