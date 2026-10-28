# GeoSentinel AI

GeoSentinel AI is a local-first academic prototype for analyzing user-provided imagery, reviewing object-detection candidates, managing restricted zones, and exporting potential alert records. It does not provide a live satellite feed. Detection candidates are not confirmed events and require human review.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- Windows PowerShell, macOS, or Linux
- YOLO11 nano weights (`yolo11n.pt`) for inference. Install backend requirements and start the API with network access once; Ultralytics downloads the official checkpoint on first load. The API reports model unavailability honestly if weights or dependencies are missing. Weight files are ignored by Git.

## Run the backend (PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Open API docs at <http://localhost:8001/docs>. The backend stores SQLite data, uploads, and annotated outputs in `backend/data` by default. Set `GEOSENTINEL_DATA_DIR` or `GEOSENTINEL_DATABASE_URL` to change local storage. The app uses CPU by default. For CUDA, install the PyTorch build appropriate to your CUDA environment before installing Ultralytics.

## Run the frontend

In a second PowerShell window, from the project root:

```powershell
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api`, `/docs`, and `/openapi.json` to the backend at <http://127.0.0.1:8001>. Set `GEOSENTINEL_API_TARGET` to change the proxy target, or set `VITE_API_BASE_URL` to use a direct API URL in another deployment.

## Current workflow

1. Upload JPG, PNG, BMP, WEBP, TIFF, MP4, AVI, MOV, MKV, or WEBM (500 MB default maximum).
2. The API persists an analysis with a queued state, then runs inference in a background thread.
3. Images produce stored YOLO detections and an annotated image. Videos are sampled at every fifth frame, capped at 3,000 frames, with ByteTrack IDs when available.
4. Applicable active pixel zones can generate potential zone-entry review records using each box's bottom-center. Pixel zones are image-specific; no real-world coordinates are fabricated.
5. Review alert status, export actual alert rows as CSV, and inspect analyses in history.

## Configuration and limitations

Copy `backend/.env.example` to `backend/.env`. Environment variables configure storage, SQLite URL, model weights, upload/frame limits, and CORS origins. Local mode has no accounts or multi-user authorization. GeoTIFF uploads are accepted as imagery, but CRS-aware coordinate conversion and geographic incident generation are not implemented yet; the map shows real OSM street tiles and applicable saved geographic polygons, with no fabricated incident pins. GeoJSON export currently reports that pixel-only results are omitted and returns an empty feature collection because geographic incident positions are not yet stored. Video playback/frame retrieval and processed video export are also not yet implemented. These boundaries are reflected in the UI and must not be treated as completed functionality.

The app is an academic prototype and has not had a production security review. Use only files you are authorized to analyze. Avoid exposing the local API to untrusted networks.
