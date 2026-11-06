# Lifora — Full-Stack Integration Bundle

This bundle adds the **FastAPI backend, MongoDB database setup, caregiver dashboard, report-upload integration, device-ingest endpoint, structured symptom/history records, medication events, rule alerts, and a RAG/LLM bridge** to the existing `HealthCopilot` project.

It is designed to be merged into the project you already have. It deliberately does **not** replace your verified `rag-llm` implementation or overwrite its existing scripts.

## Architecture included

```text
ESP32 sensors + medicine-box IR
              │  authenticated JSON
              ▼
        FastAPI backend ─────────────── React caregiver dashboard
              │                                  │
              ├── MongoDB: vitals, symptoms,     └── login, reports, alerts,
              │   history, reports, alerts          history and explanations
              │
              ├── Independent demo-rule alerts (not controlled by the LLM)
              │
              └── Existing local RAG pipeline
                    PDF → PyMuPDF → chunks → MiniLM embeddings → FAISS
                    patient report search + separate trusted-reference search
                                      │
                                      └── optional Groq explanation
                                          (disabled by default)
```

## Before you start

- Keep your existing `rag-llm/` folder and `.venv`; the backend calls its existing ingestion, chunking, indexing, and retrieval code.
- Use synthetic/demo records only while testing. The dashboard labels caregiver-entered symptoms and history as unverified.
- Groq/cloud LLM calls are disabled by default. Enable them only for the verified synthetic demo until you have completed privacy, consent, security, and legal review.
- The current experimental ML models are **not automatically activated**. The exact artifact paths and feature-preprocessing schemas must be confirmed before wiring them into an API. Alerts in this bundle come from independent configurable demo rules, not the LLM or experimental ML.
- This is a final-year project prototype, not a medical device, diagnosis service, or clinically validated alarm system.

---

## 1. Merge the bundle into your existing project

1. Download and extract `lifora_full_stack_bundle.zip`.
2. Copy its contents into your existing project root:

   `D:\FINAL YEAR PROJECT\healthcopilot\HealthCopilot`

3. Choose **merge/keep existing files** if Windows asks. Do not replace the existing `rag-llm/` folder.
4. Confirm the root now contains these new items:

   ```text
   HealthCopilot/
   ├── .env.example
   ├── .gitignore
   ├── docker-compose.yml
   ├── README_SETUP.md
   ├── backend/
   ├── frontend/
   ├── scripts/
   └── rag-llm/                 # your existing working RAG project; keep it
   ```

If your actual project root differs, use that root for every command below.

## 2. Configure environment variables

Open PowerShell at the project root:

```powershell
cd "D:\FINAL YEAR PROJECT\healthcopilot\HealthCopilot"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

**Important:** your existing RAG setup may already have a `.env` containing `GROQ_API_KEY`. Do not overwrite it. If `.env` already exists, keep its current values and add only the missing settings from `.env.example`; do not duplicate keys. Change at least these values:

```dotenv
JWT_SECRET_KEY=PASTE_A_RANDOM_SECRET_HERE
BOOTSTRAP_ADMIN_USERNAME=caregiver@example.com
BOOTSTRAP_ADMIN_PASSWORD=CHOOSE_YOUR_OWN_STRONG_LOCAL_PASSWORD
DEVICE_INGEST_TOKEN=CHOOSE_A_DIFFERENT_RANDOM_DEVICE_TOKEN
```

Generate a random JWT secret without sending it anywhere:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Copy that output directly into your local `.env` file. Do not send the secret, device token, or Groq API key in chat or commit `.env` to Git.

For the initial safe demo, leave these values as shown:

```dotenv
ALLOW_CLOUD_LLM=false
ALLOW_REAL_DATA_TO_CLOUD_LLM=false
ALLOW_REAL_REPORT_UPLOADS=false
SEED_DEMO_DATA=true
```

If you already have a Groq key, place it in the private `.env` file as `GROQ_API_KEY=...`. Keep the cloud LLM disabled until the local app is working. Do not use real identifying medical records with a free hosted API without completing privacy and security review.

## 3. Start MongoDB

### Option A — Local MongoDB with Docker

Install Docker Desktop if it is not already installed, then run from the project root:

```powershell
docker compose up -d mongodb
docker compose ps
```

The database listens only on `127.0.0.1:27017` and persists in the `lifora_mongo_data` Docker volume. The API creates the database collections and indexes at startup.

### Option B — MongoDB Atlas

If Docker is not available, create a MongoDB cluster and database user, allow access only from your development IP, and copy the cluster connection URI into `MONGO_URI` in `.env`. Do not commit the URI. Free-tier availability and limits can change. Use synthetic data only for the demo.

The default local setting is:

```dotenv
MONGO_URI=mongodb://127.0.0.1:27017
MONGO_DB=lifora_db
```

## 4. Install backend dependencies in your existing `.venv`

Open a new PowerShell terminal at the project root:

```powershell
cd "D:\FINAL YEAR PROJECT\healthcopilot\HealthCopilot"
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/requirements-rag.txt
```

The second requirements file uses the same libraries as the existing RAG pipeline. Since your current environment already loads MiniLM/FAISS, pip should reuse installed compatible packages where possible. Do not create another virtual environment unless the existing one is broken.

If the project already has a `rag-llm\requirements.txt`, install it only if your current environment is missing its dependencies:

```powershell
python -m pip install -r rag-llm/requirements.txt
```

## 5. Prepare the trusted-reference index

This is needed for the full dual-RAG explanation path. It uses the existing locally curated MedlinePlus reference summaries and keeps the reference index separate from patient reports.

From the project root, with `.venv` active:

```powershell
python rag-llm/src/trusted_reference_rag.py --self-test
python rag-llm/src/trusted_reference_rag.py build
python rag-llm/src/trusted_reference_rag.py verify
```

Expected final marker:

```text
VERIFICATION: PASS
```

The build may load the already-used `sentence-transformers/all-MiniLM-L6-v2` model. It does not make a Groq request.

## 6. Start the FastAPI backend

In the backend terminal:

```powershell
cd "D:\FINAL YEAR PROJECT\healthcopilot\HealthCopilot"
.\.venv\Scripts\Activate.ps1
Set-Location backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On first startup, the backend:

- Connects to MongoDB and creates indexes.
- Seeds a fictional `demo_patient` profile and synthetic dashboard records.
- Creates the caregiver/admin account from `.env` if it does not already exist.
- Exposes the interactive API docs.

Open these in a browser:

- API docs: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/api/v1/health`

If startup reports that MongoDB is unreachable, go back to Step 3. If it reports that the bootstrap password must be changed, edit `.env` and restart the backend. The initial account is only created once; changing the password in `.env` after it has already been seeded does not automatically reset the stored password.

## 7. Start the caregiver dashboard

Install a current Node.js LTS release if `node` and `npm` are not available. Open another terminal:

```powershell
cd "D:\FINAL YEAR PROJECT\healthcopilot\HealthCopilot\frontend"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Open `http://127.0.0.1:5173` and sign in using the username/password you set in the project-root `.env`.

The dashboard contains:

- Overview of latest HR, SpO₂, temperature and open rule alerts.
- Manual reading entry for local testing.
- Structured caregiver symptom records with severity and onset.
- Synthetic PDF upload and re-indexing through your existing RAG scripts.
- Structured history and medication entries, explicitly marked unverified.
- Medicine-box event logging.
- Optional evidence-linked RAG/LLM explanations.
- System status for database, RAG, cloud configuration and ML integration.

## 8. Run local regression tests and verify the device-ingest path

From a third terminal at the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s backend/tests -v
```

Expected: five local checks pass (password hashing, schema validation and independent alert rules). These do not require a MongoDB connection.

Keep MongoDB and the backend running, then run:

```powershell
python scripts/smoke_test.py
```

Expected final marker:

```text
SMOKE TEST: PASS
```

Send a normal synthetic sensor packet:

```powershell
python scripts/simulate_device.py --patient-id demo_patient
```

Exercise the possible-fall alert path (synthetic test only):

```powershell
python scripts/simulate_device.py --patient-id demo_patient --fall
```

Then refresh the dashboard. A possible-fall alert is generated by the independent rule engine. The LLM is not involved in generating or dismissing it.

The device endpoint is:

`POST http://127.0.0.1:8000/api/v1/devices/ingest`

It expects JSON containing `patient_id`, `device_id`, `heart_rate_bpm`, `spo2_percent`, `temperature_c`, `fall_detected`, `medicine_box_open`, and the `X-Device-Token` header. Keep the device token private. For real network deployment, use HTTPS, device identity/rotation, replay protection and proper network access controls; a token embedded in firmware can be extracted.

## 9. Test report upload and retrieval

1. Open the dashboard and select **Medical reports**.
2. Upload one of your existing synthetic sample PDFs from `rag-llm/sample_reports/`.
3. The backend validates the PDF, stores a local copy, calls `ingest_pdf.py`, `chunk_documents.py`, and `build_index.py`, and records the indexing status in MongoDB.
4. Confirm the report shows `indexed`.
5. Ask an evidence-linked question from the Overview tab after Step 10 is complete.

Only text-based PDFs are supported. Scanned PDFs requiring OCR are rejected. The backend uses a per-patient in-process lock while indexing; it is intended for one local development API process, not multiple production workers.

By default, uploads are **synthetic-only**. Do not set `ALLOW_REAL_REPORT_UPLOADS=true` until authentication/ownership, secure storage, backups, audit logging, retention, consent and applicable privacy requirements have been reviewed. Enabling real local uploads does not automatically enable cloud sharing.

## 10. Enable the Groq explanation path for synthetic demo data

Only after Steps 1–9 work:

1. Add your Groq API key privately to `.env`.
2. Confirm `GROQ_MODEL` is available for your account/provider tier. Model availability and free-tier limits can change.
3. Set:

   ```dotenv
   ALLOW_CLOUD_LLM=true
   ALLOW_REAL_DATA_TO_CLOUD_LLM=false
   ALLOW_REAL_REPORT_UPLOADS=false
   ```

4. Restart the backend and ask a question about the synthetic `demo_patient` reports.

The API checks that the report extraction files in `demo_patient` contain the synthetic test marker before allowing cloud calls. Patient-report evidence and trusted medical references are retrieved separately, and the prompt distinguishes them from caregiver-entered symptoms/history. Citation-format validation is a basic check only; it cannot prove semantic grounding or clinical correctness.

If the trusted-reference index is not built, the response will be marked `patient_report_rag_only` instead of claiming dual-RAG evidence.

## 11. API routes included

| Feature | Route |
|---|---|
| Health | `GET /api/v1/health` |
| Login / current user | `POST /api/v1/auth/login`, `GET /api/v1/auth/me` |
| Patient profiles | `GET /api/v1/patients`, `POST /api/v1/patients`, `GET /api/v1/patients/{id}` |
| Dashboard summary | `GET /api/v1/patients/{id}/dashboard` |
| Vitals | `GET/POST /api/v1/patients/{id}/vitals` |
| Device ingestion | `POST /api/v1/devices/ingest` |
| Symptoms | `GET/POST /api/v1/patients/{id}/symptoms` |
| PDF reports | `GET/POST /api/v1/patients/{id}/reports` |
| Report download | `GET /api/v1/patients/{id}/reports/{report_id}/download` |
| Medical history | `GET/POST /api/v1/patients/{id}/history` |
| Medication records/events | `/api/v1/patients/{id}/medications` and related event routes |
| Alerts | `GET /api/v1/patients/{id}/alerts`, `POST .../{alert_id}/acknowledge` |
| RAG explanation | `POST /api/v1/patients/{id}/explain` |
| System status | `GET /api/v1/system/status` |

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'pymongo'` or `fastapi`

Activate the project's `.venv` and rerun:

```powershell
python -m pip install -r backend/requirements.txt
```

### MongoDB connection error

Run `docker compose ps`, verify the `mongodb` container is healthy, and check `MONGO_URI` in `.env`. If using Atlas, verify the IP allowlist and database-user credentials.

### `GROQ_API_KEY is not configured` or cloud calls are blocked

This is expected until the private key is configured and `ALLOW_CLOUD_LLM=true`. Never paste the key into the source code or chat. Real patient data remains blocked by default.

### Trusted reference index unavailable

Run the `trusted_reference_rag.py build` and `verify` commands from Step 5. A patient-only explanation must not be described as dual-RAG.

### Report upload succeeded but indexing failed

Open the report card to inspect `index_error`, then check the backend terminal. Confirm the existing `rag-llm/src/ingest_pdf.py`, `chunk_documents.py`, and `build_index.py` scripts exist and run successfully from the project root. Keep your existing versions; do not replace them with unrelated versions.

### Frontend says it cannot connect

Make sure the backend is running on port 8000, the frontend uses `http://127.0.0.1:8000/api/v1`, and `CORS_ORIGINS` includes both `http://localhost:5173` and `http://127.0.0.1:5173`. Restart the backend after `.env` changes.

### Login fails after changing `.env`

The demo user is inserted only when first seeded. If the user already exists, the stored password is not overwritten on every restart. Reset the user through a controlled local database operation or use a fresh demo database; do not make password hashes public.

---

## Current boundaries — be transparent in your report

1. The app is a prototype; no clinical or regulatory validation is claimed.
2. Rule thresholds are configurable demonstration values, not clinically validated alert thresholds.
3. The existing ML artifact files and exact feature schemas were not part of this integration bundle, so ML inference is deliberately reported as `not_connected`/`disabled` rather than guessed.
4. MongoDB stores the structured records. PDF files and FAISS indexes are stored locally under the project folder, not inside MongoDB GridFS.
5. Text-based PDF extraction only; OCR, production malware scanning and multi-process index publication locks are not included.
6. The local device token is suitable only for a controlled prototype. Use authenticated device identity and HTTPS before deployment across networks.
7. Groq/cloud use is opt-in and synthetic-only by default. Do not transmit real medical data without appropriate consent, privacy/security controls and legal review.
