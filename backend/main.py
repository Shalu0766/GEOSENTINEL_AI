from __future__ import annotations

import csv
import io
import json
import os
import threading
import uuid
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

DATA = Path(os.getenv("GEOSENTINEL_DATA_DIR", "./data")).resolve()
UPLOADS, RESULTS = DATA / "uploads", DATA / "results"
UPLOADS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
DB_URL = os.getenv("GEOSENTINEL_DATABASE_URL", f"sqlite:///{DATA / 'geosentinel.db'}")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
MAX_MB = int(os.getenv("GEOSENTINEL_MAX_UPLOAD_MB", "500"))
MAX_FRAMES = int(os.getenv("GEOSENTINEL_MAX_VIDEO_FRAMES", "3000"))
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase): pass


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    stored_path: Mapped[str] = mapped_column(String)
    media_type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=.35)
    iou: Mapped[float] = mapped_column(Float, default=.45)
    model_name: Mapped[str] = mapped_column(String, default="yolo11n.pt")
    tracking: Mapped[bool] = mapped_column(Boolean, default=False)
    zones_json: Mapped[str] = mapped_column(Text, default="[]")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    detections: Mapped[list["Detection"]] = relationship(cascade="all, delete-orphan")
    incidents: Mapped[list["Incident"]] = relationship(cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String, index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float); y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float); y2: Mapped[float] = mapped_column(Float)
    frame_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Zone(Base):
    __tablename__ = "zones"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String, default="pixel")
    geometry_json: Mapped[str] = mapped_column(Text)
    source_filename: Mapped[str] = mapped_column(String, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id", ondelete="CASCADE"))
    zone_id: Mapped[int] = mapped_column(ForeignKey("zones.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String, default="zone_entry")
    review_status: Mapped[str] = mapped_column(String, default="pending", index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


Base.metadata.create_all(engine)
app = FastAPI(title="GeoSentinel AI", version="1.0.0", description="Local imagery analysis and geospatial review API")
origins = os.getenv("GEOSENTINEL_CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def model_status():
    try:
        from ultralytics import YOLO
        model = YOLO(os.getenv("GEOSENTINEL_MODEL", "yolo11n.pt"))
        return True, model
    except Exception:
        return False, None


def process_analysis(analysis_id: str):
    db = SessionLocal()
    try:
        a = db.get(Analysis, analysis_id)
        if not a: return
        a.status, a.progress = "running", 3; db.commit()
        available, model = model_status()
        if not available:
            a.status, a.error, a.completed_at = "failed", "YOLO model unavailable. Install backend requirements and ensure model weights can be loaded.", now(); db.commit(); return
        ext = Path(a.filename).suffix.lower()
        if ext in IMAGE_EXT:
            from PIL import Image
            with Image.open(a.stored_path) as im: im.verify()
            result = model.predict(a.stored_path, conf=a.confidence, iou=a.iou, verbose=False)[0]
            boxes = result.boxes
            for i in range(len(boxes)):
                xy = boxes.xyxy[i].tolist(); cls = int(boxes.cls[i].item())
                db.add(Detection(analysis_id=a.id, label=str(result.names.get(cls, cls)), confidence=float(boxes.conf[i].item()), x1=xy[0], y1=xy[1], x2=xy[2], y2=xy[3]))
            out = RESULTS / f"{a.id}.jpg"
            result.plot()  # ndarray, annotated with the actual model output
            import cv2
            cv2.imwrite(str(out), result.plot())
            a.progress = 85; db.commit()
            _evaluate_zones(db, a)
            a.status, a.progress, a.completed_at = "completed", 100, now(); db.commit()
        else:
            import cv2
            cap = cv2.VideoCapture(a.stored_path)
            if not cap.isOpened(): raise ValueError("Video could not be opened")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            limit = min(total, MAX_FRAMES) if total else MAX_FRAMES
            frame_no = 0
            while frame_no < limit:
                ok, frame = cap.read()
                if not ok: break
                if frame_no % 5 == 0:
                    if a.tracking:
                        result = model.track(frame, conf=a.confidence, iou=a.iou, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
                    else:
                        result = model.predict(frame, conf=a.confidence, iou=a.iou, verbose=False)[0]
                    if result.boxes is not None:
                        for i in range(len(result.boxes)):
                            b = result.boxes; xy = b.xyxy[i].tolist(); cls = int(b.cls[i].item())
                            tid = int(b.id[i].item()) if b.id is not None else None
                            db.add(Detection(analysis_id=a.id, label=str(result.names.get(cls, cls)), confidence=float(b.conf[i].item()), x1=xy[0], y1=xy[1], x2=xy[2], y2=xy[3], frame_number=frame_no, track_id=tid))
                    cv2.imwrite(str(RESULTS / f"{a.id}-frame-{frame_no}.jpg"), result.plot())
                frame_no += 1
                if frame_no % 100 == 0:
                    a.progress = min(90, int(frame_no / max(limit, 1) * 90)); db.commit()
            cap.release(); _evaluate_zones(db, a)
            a.status, a.progress, a.completed_at = "completed", 100, now(); db.commit()
    except Exception as exc:
        db.rollback(); a = db.get(Analysis, analysis_id)
        if a: a.status, a.error, a.completed_at = "failed", str(exc)[:500], now(); db.commit()
    finally: db.close()


def _evaluate_zones(db: Session, analysis: Analysis):
    try:
        from shapely.geometry import Point, Polygon
    except ImportError: return
    zones = [z for z in db.query(Zone).filter(Zone.active.is_(True)).all() if z.mode == "pixel" and (not z.source_filename or z.source_filename == analysis.filename)]
    selected = set(json.loads(analysis.zones_json or "[]"))
    zones = [z for z in zones if z.id in selected]
    for d in db.query(Detection).filter_by(analysis_id=analysis.id).all():
        point = Point((d.x1 + d.x2) / 2, d.y2)
        for z in zones:
            poly = Polygon(json.loads(z.geometry_json))
            if poly.is_valid and poly.covers(point):
                db.add(Incident(analysis_id=analysis.id, detection_id=d.id, zone_id=z.id))
    db.flush()


@app.get("/api/v1/health")
def health():
    ok, _ = model_status()
    return {"backend": "online", "database": "online", "storage": UPLOADS.exists() and RESULTS.exists(), "model_available": ok, "model_name": os.getenv("GEOSENTINEL_MODEL", "yolo11n.pt"), "max_upload_mb": MAX_MB, "max_video_frames": MAX_FRAMES}

@app.get("/api/v1/settings")
def settings(): return health() | {"docs_url": "/docs", "coordinate_modes": ["geographic", "pixel"]}

@app.get("/api/v1/models")
def models():
    available, model = model_status()
    return [{"name": os.getenv("GEOSENTINEL_MODEL", "yolo11n.pt"), "available": available, "classes": sorted(model.names.values()) if available else [], "device": "auto" if available else "unavailable", "tracking": available}]

@app.post("/api/v1/analyses")
async def create_analysis(file: UploadFile = File(...), confidence: float = Form(.35), iou: float = Form(.45), tracking: bool = Form(False), zones: str = Form("[]")):
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1: raise HTTPException(422, "Confidence and IoU must be between 0 and 1")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT: raise HTTPException(415, "Unsupported file type")
    aid = str(uuid.uuid4()); target = UPLOADS / f"{aid}{ext}"
    size = 0
    try:
        with target.open("wb") as dst:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024: raise HTTPException(413, f"File exceeds {MAX_MB} MB limit")
                dst.write(chunk)
    except Exception:
        target.unlink(missing_ok=True); raise
    if size == 0: target.unlink(missing_ok=True); raise HTTPException(422, "Empty file")
    safe_name = Path(file.filename or "upload").name[:200]
    media = "video" if ext in VIDEO_EXT else ("geotiff" if ext in {".tif", ".tiff"} else "image")
    db = SessionLocal(); a = Analysis(id=aid, filename=safe_name, stored_path=str(target), media_type=media, confidence=confidence, iou=iou, tracking=tracking, zones_json=zones)
    db.add(a); db.commit(); db.close()
    threading.Thread(target=process_analysis, args=(aid,), daemon=True).start()
    return {"id": aid, "status": "queued", "filename": safe_name}

def serialize_analysis(db, a):
    return {"id": a.id, "filename": a.filename, "media_type": a.media_type, "status": a.status, "progress": a.progress, "confidence": a.confidence, "iou": a.iou, "model_name": a.model_name, "tracking": a.tracking, "error": a.error, "created_at": a.created_at.isoformat(), "completed_at": a.completed_at.isoformat() if a.completed_at else None, "detection_count": db.query(Detection).filter_by(analysis_id=a.id).count(), "alert_count": db.query(Incident).filter_by(analysis_id=a.id).count()}

@app.get("/api/v1/analyses")
def analyses():
    db=SessionLocal(); rows=db.query(Analysis).order_by(Analysis.created_at.desc()).all(); out=[serialize_analysis(db,a) for a in rows]; db.close(); return out

@app.get("/api/v1/analyses/{analysis_id}")
@app.get("/api/v1/analyses/{analysis_id}/status")
def analysis(analysis_id: str):
    db=SessionLocal(); a=db.get(Analysis,analysis_id)
    if not a: db.close(); raise HTTPException(404,"Analysis not found")
    out=serialize_analysis(db,a); db.close(); return out

@app.post("/api/v1/analyses/{analysis_id}/retry")
def retry(analysis_id: str):
    db=SessionLocal(); a=db.get(Analysis,analysis_id)
    if not a: db.close(); raise HTTPException(404,"Analysis not found")
    if a.status != "failed": db.close(); raise HTTPException(409,"Only failed analyses can be retried")
    a.status="queued"; a.progress=0; a.error=None; db.query(Detection).filter_by(analysis_id=a.id).delete(); db.query(Incident).filter_by(analysis_id=a.id).delete(); db.commit(); db.close()
    threading.Thread(target=process_analysis,args=(analysis_id,),daemon=True).start(); return {"status":"queued"}

@app.delete("/api/v1/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    db=SessionLocal(); a=db.get(Analysis,analysis_id)
    if not a: db.close(); raise HTTPException(404,"Analysis not found")
    Path(a.stored_path).unlink(missing_ok=True); (RESULTS/f"{a.id}.jpg").unlink(missing_ok=True); db.delete(a); db.commit(); db.close(); return {"deleted":True}

@app.get("/api/v1/analyses/{analysis_id}/detections")
def detections(analysis_id: str):
    db=SessionLocal(); rows=db.query(Detection).filter_by(analysis_id=analysis_id).all(); out=[{k:getattr(d,k) for k in ("id","label","confidence","x1","y1","x2","y2","frame_number","track_id")} for d in rows]; db.close(); return out

@app.get("/api/v1/analyses/{analysis_id}/frames/{frame_number}")
def analysis_frame(analysis_id: str, frame_number: int):
    p = RESULTS / f"{analysis_id}-frame-{frame_number}.jpg"
    if not p.exists(): raise HTTPException(404, "Annotated sampled frame unavailable")
    return FileResponse(p, media_type="image/jpeg", filename=f"{analysis_id}-frame-{frame_number}.jpg")

@app.get("/api/v1/analyses/{analysis_id}/tracks")
def tracks(analysis_id: str):
    db=SessionLocal(); rows=db.query(Detection).filter(Detection.analysis_id==analysis_id,Detection.track_id.is_not(None)).all(); out={}
    for d in rows: out.setdefault(str(d.track_id),{"track_id":d.track_id,"label":d.label,"observations":0,"frames":[]})["observations"]+=1; out[str(d.track_id)]["frames"].append(d.frame_number)
    db.close(); return list(out.values())

@app.get("/api/v1/analyses/{analysis_id}/annotated")
def annotated(analysis_id: str):
    p=RESULTS/f"{analysis_id}.jpg"
    if not p.exists(): raise HTTPException(404,"Annotated image unavailable")
    return FileResponse(p,media_type="image/jpeg",filename=f"{analysis_id}-annotated.jpg")

class ZoneInput(BaseModel):
    name: str = Field(min_length=1,max_length=120); description: str = ""; mode: str = "pixel"; geometry: list[list[float]]; source_filename: str = ""; active: bool = True

@app.get("/api/v1/zones")
def list_zones():
    db=SessionLocal(); zs=db.query(Zone).order_by(Zone.id.desc()).all(); out=[zone_dict(z) for z in zs]; db.close(); return out

def zone_dict(z): return {"id":z.id,"name":z.name,"description":z.description,"mode":z.mode,"geometry":json.loads(z.geometry_json),"source_filename":z.source_filename,"active":z.active}

@app.post("/api/v1/zones")
def create_zone(payload: ZoneInput):
    if payload.mode not in ("pixel","geographic"): raise HTTPException(422,"Mode must be pixel or geographic")
    if len(payload.geometry)<3: raise HTTPException(422,"Polygon requires at least three points")
    if payload.mode=="geographic" and any(abs(p[0])>180 or abs(p[1])>90 for p in payload.geometry): raise HTTPException(422,"Geographic points must be [longitude, latitude]")
    coords=payload.geometry[:]
    if coords[0]!=coords[-1]: coords.append(coords[0])
    try:
        from shapely.geometry import Polygon
        if not Polygon(coords).is_valid: raise ValueError()
    except Exception: raise HTTPException(422,"Invalid polygon geometry")
    db=SessionLocal(); z=Zone(name=payload.name,description=payload.description,mode=payload.mode,geometry_json=json.dumps(coords),source_filename=payload.source_filename,active=payload.active); db.add(z); db.commit(); db.refresh(z); out=zone_dict(z); db.close(); return out

@app.patch("/api/v1/zones/{zone_id}")
def update_zone(zone_id:int,payload:dict):
    db=SessionLocal(); z=db.get(Zone,zone_id)
    if not z: db.close(); raise HTTPException(404,"Zone not found")
    for key in ("name","description","active","source_filename"):
        if key in payload: setattr(z,key,payload[key])
    if "geometry" in payload: z.geometry_json=json.dumps(payload["geometry"])
    db.commit(); out=zone_dict(z); db.close(); return out

@app.delete("/api/v1/zones/{zone_id}")
def delete_zone(zone_id:int):
    db=SessionLocal(); z=db.get(Zone,zone_id)
    if not z: db.close(); raise HTTPException(404,"Zone not found")
    db.query(Incident).filter_by(zone_id=zone_id).delete(); db.delete(z); db.commit(); db.close(); return {"deleted":True}

@app.get("/api/v1/alerts")
def alerts(status:Optional[str]=None):
    db=SessionLocal(); q=db.query(Incident).order_by(Incident.updated_at.desc())
    if status: q=q.filter(Incident.review_status==status)
    rows=q.all(); out=[]
    for i in rows:
        d=db.get(Detection,i.detection_id); z=db.get(Zone,i.zone_id); a=db.get(Analysis,i.analysis_id)
        out.append({"id":i.id,"analysis_id":i.analysis_id,"filename":a.filename if a else "","detection_id":i.detection_id,"label":d.label if d else "","confidence":d.confidence if d else 0,"zone":z.name if z else "","event_type":i.event_type,"review_status":i.review_status,"notes":i.notes,"frame_number":d.frame_number if d else None,"updated_at":i.updated_at.isoformat()})
    db.close(); return out

@app.patch("/api/v1/alerts/{alert_id}")
def review_alert(alert_id:int,payload:dict):
    db=SessionLocal(); i=db.get(Incident,alert_id)
    if not i: db.close(); raise HTTPException(404,"Alert not found")
    status=payload.get("review_status",i.review_status)
    if status not in ("pending","reviewed","false_positive"): db.close(); raise HTTPException(422,"Invalid review status")
    i.review_status=status; i.notes=str(payload.get("notes",i.notes))[:2000]; i.updated_at=now(); db.commit(); db.close(); return {"updated":True}

@app.get("/api/v1/alerts/export.csv")
def export_csv():
    rows=alerts(); buf=io.StringIO(); fields=["id","analysis_id","filename","label","confidence","zone","event_type","review_status","notes","frame_number"]
    writer=csv.DictWriter(buf,fieldnames=fields); writer.writeheader(); writer.writerows([{k:r.get(k) for k in fields} for r in rows]); return StreamingResponse(iter([buf.getvalue()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=geosentinel-alerts.csv"})

@app.get("/api/v1/alerts/export.geojson")
def export_geojson(): return {"type":"FeatureCollection","features":[],"note":"Pixel-space incidents have no geographic coordinates and are omitted."}
