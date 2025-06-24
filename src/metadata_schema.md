# ModelFloat MVP — Metadata Registry Schema

**Purpose:**  
Define the minimal schema for tracking models, fragments, and storage locations in the MVP.

---

## Model Metadata

| Field         | Type    | Description                          |
|---------------|---------|--------------------------------------|
| model_id      | string  | Unique identifier for the model      |
| name          | string  | Human-readable model name            |
| description   | string  | Short description of the model       |
| framework     | string  | Model framework (e.g., pytorch)      |
| size_bytes    | int     | Model size in bytes                  |
| created_at    | string  | ISO timestamp                        |

---

## Fragment Metadata

| Field         | Type    | Description                          |
|---------------|---------|--------------------------------------|
| fragment_id   | string  | Unique identifier for the fragment   |
| model_id      | string  | Parent model identifier              |
| index         | int     | Fragment index (order)               |
| size_bytes    | int     | Fragment size in bytes               |
| checksum      | string  | SHA256 or similar hash               |
| storage       | list    | List of storage locations            |

---

## Storage Location Metadata

| Field         | Type    | Description                          |
|---------------|---------|--------------------------------------|
| fragment_id   | string  | Fragment identifier                  |
| provider      | string  | Storage provider name                |
| location      | string  | Provider-specific location/id/url    |
| uploaded_at   | string  | ISO timestamp                        |

---

## Example (JSON)

```json
{
  "models": [
    {
      "model_id": "llama-2-7b",
      "name": "LLaMA 2 7B",
      "description": "Meta's LLaMA 2 7B parameter model",
      "framework": "pytorch",
      "size_bytes": 13476838400,
      "created_at": "2025-06-24T13:00:00Z"
    }
  ],
  "fragments": [
    {
      "fragment_id": "frag-001",
      "model_id": "llama-2-7b",
      "index": 0,
      "size_bytes": 100663296,
      "checksum": "abc123...",
      "storage": [
        {
          "provider": "mega",
          "location": "https://mega.nz/file/...",
          "uploaded_at": "2025-06-24T13:01:00Z"
        }
      ]
    }
  ]
}
```

---

*This schema will be used for the MVP metadata registry (file or SQLite).*
