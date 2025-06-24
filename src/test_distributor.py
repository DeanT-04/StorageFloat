# test_distributor.py
"""
Unit tests for ModelFloat MVP distributor.py
"""

from distributor import FragmentDistributor

class DummyAdapter:
    def __init__(self, name):
        self.name = name
        self.uploaded = []
    def upload_fragment(self, fragment_path):
        self.uploaded.append(fragment_path)
        return f"{self.name}_loc_{fragment_path}"

def test_distribute_fragment_basic():
    mega = DummyAdapter("mega")
    gdrive = DummyAdapter("gdrive")
    ipfs = DummyAdapter("ipfs")
    distributor = FragmentDistributor({
        "mega": mega,
        "gdrive": gdrive,
        "ipfs": ipfs
    })
    locations = distributor.distribute_fragment("frag1.bin", redundancy=2)
    assert len(locations) == 2
    providers = {loc["provider"] for loc in locations}
    assert providers.issubset({"mega", "gdrive", "ipfs"})
    for loc in locations:
        assert loc["location"].endswith("frag1.bin")

def test_distribute_fragment_redundancy_greater_than_providers():
    mega = DummyAdapter("mega")
    gdrive = DummyAdapter("gdrive")
    distributor = FragmentDistributor({
        "mega": mega,
        "gdrive": gdrive
    })
    locations = distributor.distribute_fragment("frag2.bin", redundancy=5)
    assert len(locations) == 2  # Only two providers available

def test_distribute_fragment_no_providers():
    distributor = FragmentDistributor({})
    locations = distributor.distribute_fragment("frag3.bin", redundancy=2)
    assert locations == []
