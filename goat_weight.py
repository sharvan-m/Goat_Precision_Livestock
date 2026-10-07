from pathlib import Path
import base64, io, os, threading
import numpy as np, cv2, joblib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    from typing import List
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

ROOT = r"D:\weight_detect"
MODEL_DIR = os.path.join(ROOT, "goat_metric_yw")
Z_TESTED = (90.0, 145.0)      # camera distance range (cm) covered by the training photos
Z_HARD = (60.0, 300.0)        # outside this the depth reading is not trusted at all
MIN_GOAT_W_PX = 385           # smallest goat width (px) tested in the far-distance simulation
WEIGHT_RANGE = (12.0, 90.0)   # trained on 14.8-81.4 kg
TAPE_BAND = 0.20              # 80% of goats within +-19% on the 24 Jining goats
NOTES = ["Not for drug dosing. Use a scale weight for dosing decisions.",
         "Model trained on 24 Jining Qing goats (one breed, 15-81 kg, 4-48 months); other breeds and setups are not validated."]
_lock = threading.Lock(); _state = {}

def load_model(view):
    key = "m_" + view
    if key not in _state:
        p = MODEL_DIR / ("final_area_%s.joblib" % view)
        if not os.path.exists(p): raise FileNotFoundError("model file missing: " + p + " (run fit_final_models.py)")
        _state[key] = joblib.load(p)
    return _state[key]

def _detector():
    with _lock:
        if "det" not in _state:
            import torch
            from ultralytics import SAM, YOLOWorld
            dev = 0 if torch.cuda.is_available() else "cpu"
            yp = ROOT / "yolov8s-worldv2.pt"`r`n            sp = ROOT / "mobile_sam.pt"
            yw = YOLOWorld(yp if os.path.exists(yp) else "yolov8s-worldv2.pt"); yw.set_classes(["goat"])
            sam = SAM(sp if os.path.exists(sp) else "mobile_sam.pt")
            _state["det"] = (yw, sam, dev)
    return _state["det"]

def segment_goat(rgb):
    yw, sam, dev = _detector()
    with _lock:
        r = yw.predict(rgb[..., ::-1], conf=0.15, device=dev, verbose=False)[0]
        if len(r.boxes) == 0: return None
        conf = r.boxes.conf.cpu().numpy(); i = int(conf.argmax()); box = r.boxes.xyxy[i].cpu().numpy()
        m = sam(rgb[..., ::-1], bboxes=[[float(v) for v in box]], device=dev, verbose=False)[0].masks.data[0].cpu().numpy().astype(bool)
    return m, box, float(conf[i]), int((conf >= 0.25).sum())

def measure(mask, depth, fx):
    h, w = depth.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2: return None, "empty mask"
    mm = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))); ys, xs = np.nonzero(mm)
    d = depth[mm].astype(np.float64); d = d[(d > 500) & (d < 6000)]
    if len(d) < 0.3 * len(xs): return None, "no valid depth values on the goat"
    z = float(np.median(d)); cmpx = z / 10.0 / fx
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False); pr = (pts - c) @ vt[0]
    return dict(z_cm=z / 10.0, area=len(xs) * cmpx ** 2, BL=(np.percentile(pr, 99.5) - np.percentile(pr, .5)) * cmpx,
                width_px=int(xs.max() - xs.min() + 1),
                touches=bool(xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3 or ys.max() >= h - 3)), None

def read_depth(data):
    a = np.array(Image.open(io.BytesIO(data)))
    if a.ndim != 2: raise ValueError("the depth file must be a single-channel 16-bit PNG (depth in millimetres)")
    return a.astype(np.int32)

def predict_one(model, rgb, depth, segment):
    base = dict(status="no_reading", kg=None, reasons=[], warnings=[], features=None, _mask=None)
    if rgb.shape[:2] != depth.shape:
        base["reasons"].append("depth frame size %dx%d does not match the photo %dx%d; use a depth frame aligned to the photo" % (depth.shape[1], depth.shape[0], rgb.shape[1], rgb.shape[0])); return base
    seg = segment(rgb)
    if seg is None:
        base["reasons"].append("no goat detected"); return base
    mask, box, score, n_goats = seg
    if n_goats > 1: base["warnings"].append("more than one goat may be in view; the most confident one was measured")
    f, err = measure(mask, depth, model["fx"])
    if f is None:
        base["reasons"].append(err); return base
    base["features"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in f.items()}; base["_mask"] = mask
    if f["touches"]: base["reasons"].append("the goat touches the edge of the photo; keep the whole body in frame")
    if f["width_px"] < MIN_GOAT_W_PX: base["reasons"].append("the goat is too small in the photo (%d px wide, need at least %d)" % (f["width_px"], MIN_GOAT_W_PX))
    if not (Z_HARD[0] <= f["z_cm"] <= Z_HARD[1]): base["reasons"].append("measured distance %.0f cm is not plausible for this depth sensor" % f["z_cm"])
    if not (35 <= f["BL"] <= 150): base["reasons"].append("measured body length %.0f cm is implausible; the goat may be partly hidden" % f["BL"])
    if not (Z_TESTED[0] <= f["z_cm"] <= Z_TESTED[1]): base["warnings"].append("distance %.0f cm is outside the tested %d-%d cm; scale unverified" % (f["z_cm"], *Z_TESTED))
    if base["reasons"]: return base
    kg = float(np.exp(model["coef"][0] + model["coef"][1] * np.log(f["area"])))
    base["kg"] = kg; base["status"] = "ok"; return base

def combine(preds, model):
    ok = [p for p in preds if p["kg"] is not None]
    out = dict(status="no_reading", weight_kg=None, range_kg=None, confidence="none", photos_used=len(ok), photos_total=len(preds), notes=list(NOTES), reasons=[])
    if not ok:
        out["reasons"] = sorted({r for p in preds for r in p["reasons"]}); return out
    kgs = np.array([p["kg"] for p in ok]); med = float(np.median(kgs))
    spread = float((kgs.max() - kgs.min()) / med) if len(kgs) > 1 else None
    if not (WEIGHT_RANGE[0] <= med <= WEIGHT_RANGE[1]):
        out["reasons"].append("estimated %.0f kg is outside the trained range %d-%d kg (kids, very large animals and other breeds need their own model)" % (med, *WEIGHT_RANGE)); return out
    band = model["band80_median"] if len(ok) >= 3 else model["band80_one"]
    if spread is not None: band = max(band, spread / 2)
    unverified = any("scale unverified" in w for p in ok for w in p["warnings"])
    conf = "good" if (len(ok) >= 3 and spread is not None and spread <= 0.20 and not unverified) else "rough"
    out.update(status="ok", weight_kg=round(med, 1), range_kg=[round(med * (1 - band), 1), round(med * (1 + band), 1)], band_pct=round(band * 100), confidence=conf,
               photo_spread_pct=None if spread is None else round(spread * 100))
    out["notes"] += sorted({w for p in preds for w in p["warnings"]})
    if len(ok) < len(preds): out["notes"].append("%d of %d photos could not be used" % (len(preds) - len(ok), len(preds)))
    return out

def tape_weight(hg_cm, bl_cm):
    kg = hg_cm ** 2 * bl_cm / 10840.0
    notes = ["Lambourne formula: heart girth^2 x body length / 10840 (cm, kg).", NOTES[0],
             "On 24 Jining goats it read low by about 3% (<30 kg), 7% (30-60 kg) and 10% (>60 kg); 80% of goats were within +-19%."]
    return dict(status="ok", weight_kg=round(kg, 1), range_kg=[round(kg * (1 - TAPE_BAND), 1), round(kg * (1 + TAPE_BAND), 1)], confidence="rough", notes=notes)

def overlay_b64(rgb, mask, text):
    ov = rgb.copy(); ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    sc = 1000 / ov.shape[1]; ov = cv2.resize(ov, None, fx=sc, fy=sc)
    cv2.putText(ov, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii")

def make_router(segment=None):
    seg = segment or segment_goat
    router = APIRouter(prefix="/api/weight", tags=["goat-weight"])

    @router.get("/status")
    def status():
        have = {v: (MODEL_DIR / ("final_area_%s.joblib" % v)).exists() for v in ("main", "assist", "pooled")}
        try:
            import torch; gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        return dict(models=have, gpu=gpu, weight_range_kg=list(WEIGHT_RANGE), tested_distance_cm=list(Z_TESTED), min_goat_width_px=MIN_GOAT_W_PX)

    @router.post("/predict")
    def predict(view: str = Form("pooled"), rgb: List[UploadFile] = File(...), depth: List[UploadFile] = File(...)):
        if view not in ("main", "assist", "pooled"): raise HTTPException(400, "view must be main, assist or pooled")
        if len(rgb) != len(depth) or not (1 <= len(rgb) <= 6): raise HTTPException(400, "send 1 to 6 photos, each with its own depth file")
        try: model = load_model(view)
        except FileNotFoundError as e: raise HTTPException(503, str(e))
        preds, first = [], None
        for r, d in zip(rgb, depth):
            try:
                rb, db = r.file.read(), d.file.read()
                if len(rb) > 60e6 or len(db) > 60e6: raise ValueError("file too large")
                img = np.array(Image.open(io.BytesIO(rb)).convert("RGB")); dep = read_depth(db)
            except Exception as e:
                preds.append(dict(status="no_reading", kg=None, reasons=["could not read the files: %s" % e], warnings=[], features=None, _mask=None)); continue
            p = predict_one(model, img, dep, seg)
            if first is None and p["_mask"] is not None: first = (img, p["_mask"], p)
            preds.append(p)
        res = combine(preds, model)
        res["per_photo"] = [{k: v for k, v in p.items() if k != "_mask"} for p in preds]
        res["model"] = dict(view=view, n_goats=model["n_goats"], cv_mae_one_photo_kg=round(model["cv_mae_one"], 1), cv_mae_median_kg=round(model["cv_mae_median"], 1))
        if first is not None:
            txt = ("%.0f kg (%.0f-%.0f)" % (res["weight_kg"], *res["range_kg"])) if res["status"] == "ok" else "no reading"
            res["overlay_jpeg_base64"] = overlay_b64(first[0], first[1], txt)
        return res

    @router.post("/tape")
    def tape(hg_cm: float = Form(...), bl_cm: float = Form(...)):
        if not (30 <= hg_cm <= 130 and 30 <= bl_cm <= 130): raise HTTPException(400, "heart girth and body length should be between 30 and 130 cm")
        return tape_weight(hg_cm, bl_cm)

    return router

