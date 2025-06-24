# distributor.py
"""
ModelFloat MVP — Fragment Distributor

Distributes model fragments across multiple storage providers for redundancy.
For the MVP, supports MEGA, Google Drive, and IPFS adapters.
"""

import random
from typing import List, Dict, Callable

# Provider adapter classes should be imported here in real usage
# For MVP, pass in adapter instances via the constructor

class FragmentDistributor:
    def __init__(self, providers: Dict[str, Callable]):
        """
        providers: dict mapping provider name to adapter instance
        Example: {"mega": mega_adapter, "gdrive": gdrive_adapter, "ipfs": ipfs_adapter}
        """
        self.providers = providers
        self.provider_names = list(providers.keys())

    def distribute_fragment(self, fragment_path: str, redundancy: int = 2) -> List[Dict]:
        """
        Uploads the fragment to 'redundancy' different providers.
        Returns a list of storage location dicts:
          - provider: provider name
          - location: provider-specific id/link/hash
        """
        chosen = random.sample(self.provider_names, min(redundancy, len(self.provider_names)))
        locations = []
        for name in chosen:
            adapter = self.providers[name]
            if name == "mega":
                loc = adapter.upload_fragment(fragment_path)
            elif name == "gdrive":
                loc = adapter.upload_fragment(fragment_path)
            elif name == "ipfs":
                loc = adapter.upload_fragment(fragment_path)
            else:
                continue
            if loc:
                locations.append({"provider": name, "location": loc})
        return locations
