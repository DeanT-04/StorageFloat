# ModelFloat MVP

**Free Cloud-Based Floating Storage System for Open Source AI Models**

---

## Overview

ModelFloat is a free, distributed storage and serving system for open source AI models. It uses MEGA, Google Drive, and IPFS for storage, supports model fragmentation and reassembly, and provides a REST API for model upload, listing, and inference (mocked for MVP).

**Models are never stored permanently on your desktop.**  
- All model fragments are uploaded and redundantly stored across multiple cloud providers.
- When a model is needed, fragments are securely and quickly retrieved from the cloud, reassembled in memory or temporary storage, and then removed after use.
- This ensures you can access your models from anywhere, at any time, without relying on local disk storage.

**Speed, Security, and Reliability:**  
- Fragments are distributed with redundancy for high availability.
- Retrieval is parallelized for speed.
- All transfers use secure APIs and can be encrypted.
- If a provider is unavailable, fragments are fetched from backups on other providers.

---

## Features

- Model fragmentation and reassembly
- Distributed storage across MEGA, Google Drive, and IPFS
- Redundant fragment distribution
- Metadata registry (JSON)
- REST API (FastAPI): upload, list, inference (mocked)
- Full unit test coverage
- Dockerized deployment

---

## Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd Floating_storage

# Build and run with Docker
docker build -t modelfloat .
docker run -p 8000:8000 modelfloat

# Run tests inside Docker
docker run modelfloat pytest src/
```

---

## API Endpoints

- `POST /models/upload` — Upload a model file
- `GET /models` — List all models
- `POST /models/{model_id}/inference` — Run mocked inference

---

## Roadmap Checklist

### ✅ Completed

- [x] Model fragmentation logic
- [x] MEGA, Google Drive, IPFS storage adapters
- [x] Fragment distribution and redundancy
- [x] Metadata registry (JSON)
- [x] Fragment assembler (retrieval and reassembly)
- [x] REST API (upload, list, inference [mocked])
- [x] Full unit test coverage (pytest)
- [x] Project documentation and summary
- [x] Docker container for easy install

### 🟡 In Progress / Next

- [ ] Integrate real fragment logic into API endpoints
- [ ] Update datetime handling to timezone-aware objects
- [ ] Add user authentication and access control
- [ ] Expand provider support (more storage backends)
- [ ] Production deployment scripts and monitoring

---

## Development & Testing

- All code is in `src/`
- Run tests: `pytest src/`
- See `milestones.md` and `tasks.md` for detailed progress tracking

---

## License

MIT License

---

*See SUMMARY.md for a full project overview and architecture.*
