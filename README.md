# MailParser - Shipping Email Intelligence

AI-powered shipping email classification and structured data extraction platform. Classifies emails into **tonnage**, **voyage charter (cargo\_vc)**, or **time charter (cargo\_tc)** categories and extracts all structured records - using local Python NLP only (no external LLM APIs).

---

## Architecture

```text
IME/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── classifier.py    # Hybrid classification (rules + sklearn ML)
│   │   ├── extractor.py     # Multi-record regex extraction
│   │   ├── trainer.py       # Synthetic data generation + model training
│   │   └── models.py        # Pydantic request/response schemas
│   ├── models/              # Persisted model.pkl (auto-generated)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html           # Glassmorphism UI
│   ├── style.css            # Dark theme design system
│   ├── app.js               # API client + result renderer
│   ├── schema.sql           # Database schema for Cloudflare D1
│   ├── wrangler.toml        # Cloudflare Pages / D1 binding config
│   └── functions/
│       └── api/
│           └── save.js      # Serverless function to persist data to D1
└── README.md
```

---

## Quick Start

### Backend (local)

```bash
cd backend
pip install -r requirements.txt

# Train the model (optional - auto-trains on first startup)
python -m app.trainer

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Backend (Docker)

```bash
cd backend
docker build -t mailparser .
docker run -p 8000:8000 mailparser
```

### Frontend

Open `frontend/index.html` in your browser. Update the `API_BASE` constant in `app.js` if your backend is not on `localhost:8000`.

---

## API

### `GET /health`

Health probe. Returns `{"status": "healthy"}`.

### `POST /parse-email`

**Request:**
```json
{
  "email_body": "MV SEA BRIGHT\nOPEN SINGAPORE 15 JUN\n58K DWT SUPRAMAX\nPLS REVERT WITH SUITABLE CARGO"
}
```

**Response:**
```json
{
  "success": true,
  "category": "tonnage",
  "confidence": 0.92,
  "records": [
    {
      "vessel_name": "SEA BRIGHT",
      "account_name": "",
      "open_port": "SINGAPORE",
      "open_date": "15 JUN",
      "vessel_type": "SUPRAMAX",
      "vessel_size": "58000"
    }
  ],
  "metadata": {
    "records_found": 1,
    "processing_time_ms": 12.5
  }
}
```

---

## Classification Engine

The classifier uses a **hybrid two-layer** approach:

1. **Layer 1 - Rule-based**: Weighted keyword signals (MV/DWT → tonnage, LAYCAN/VOYAGE → cargo\_vc, DELIVERY/REDELIVERY → cargo\_tc)
2. **Layer 2 - ML**: TF-IDF (n-gram 1–3) + Logistic Regression trained on ~900 synthetic samples

Both layers are blended (55% rules, 45% ML) for the final prediction.

---

## Categories

| Category    | Description                    | Key Signals                              |
|-------------|--------------------------------|------------------------------------------|
| `tonnage`   | Vessel availability position   | MV, OPEN, DWT, vessel type keywords      |
| `cargo_vc`  | Voyage charter cargo enquiry   | LOAD PORT, DISCHARGE PORT, LAYCAN, QTY   |
| `cargo_tc`  | Time charter cargo requirement | DELIVERY, REDELIVERY, DURATION, PERIOD   |

---

## Tech Stack

- **Backend**: Python 3.12, FastAPI, scikit-learn, Pydantic
- **Frontend**: Vanilla HTML/CSS/JS
- **Database**: Cloudflare D1 (Serverless SQLite)
- **ML**: TF-IDF vectorizer + Logistic Regression (local)
- **Deployment**: Railway (Monorepo backend), Cloudflare Pages (Monorepo frontend + serverless functions)
