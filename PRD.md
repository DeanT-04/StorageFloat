# Product Requirements Document (PRD): ModelFloat MVP

**Project:** ModelFloat — Free Cloud-Based Floating Storage System for Open Source AI Models  
**Version:** MVP  
**Author:** DeanT-04  
**Date:** 2025-06-24

---

## 1. Vision

Democratize access to open source AI models by providing a zero-cost, cloud-based, distributed storage and inference system, accessible via a simple API.

## 2. Goals

- Store and serve open source AI models using only free-tier cloud resources.
- Enable upload, fragmentation, distributed storage, retrieval, and assembly of models.
- Provide a REST API for model upload, listing, and inference (mocked for MVP).
- Ensure all code is well-tested, documented, and modular.

## 3. MVP Scope

- Model fragmentation and distribution (simulate with small files).
- Storage adapters for at least two free providers (e.g., MEGA, Google Drive, IPFS).
- Retrieval and assembly logic.
- Minimal metadata registry (file or SQLite).
- REST API endpoints: upload, list, inference (mocked).
- Automated and manual tests for all features.

## 4. Out of Scope

- Production-scale security, scaling, or UI.
- Full-featured user management.
- Paid/pro features.

## 5. Success Criteria

- All MVP features implemented and tested.
- All milestones tracked in `milestones.md`.
- All tasks/subtasks tracked in `tasks.md`.
- All code and docs committed to git.

---

*This PRD is a living document and will be updated throughout the project.*
