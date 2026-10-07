import os, sys, numpy as np, cv2, joblib
MODELDIR = r"D:\weight_detect\goat_metric_yw"
OUTDIR = r"D:\weight_detect\infer_out"
Z_RANGE = (90.0, 145.0)

def metric_feats(mask, depth, fx):
    h, w = depth.shape
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2: return None, "empty mask"
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); mm = lab == k
    ys, xs = np.nonzero(mm)
    d = depth[mm].astype(np.float64); d = d[(d > 500) & (d < 6000)]
    if len(d) < 0.3 * len(xs): return None, "depth missing inside the mask"
    z = float(np.median(d)); cmpx = z / 10.0 / fx
    pts = np.stack([xs, ys], 1).astype(np.float64); c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    pr, pp = (pts - c) @ vt[0], (pts - c) @ vt[1]
    f = dict(z_cm=z / 10, BL=(np.percentile(pr, 99.5) - np.percentile(pr, .5)) * cmpx,
             H=(np.percentile(pp, 99.5) - np.percentile(pp, .5)) * cmpx, area=len(xs) * cmpx ** 2,
             touches=bool(xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3 or ys.max() >= h - 3))
    return f, None

def predict_weight(model, f):
    why = []
    if f["touches"]: why.append("goat touches the image border")
    if not (Z_RANGE[0] <= f["z_cm"] <= Z_RANGE[1]): why.append("camera distance %.0f cm is outside the trained range %d-%d cm" % (f["z_cm"], *Z_RANGE))
    if not (35 <= f["BL"] <= 150): why.append("body length %.0f cm is implausible" % f["BL"])
    if why: return None, why
    X = np.array([1.0] + [np.log(f[c]) for c in model["features"]])
    return float(np.exp(X @ model["coef"])), []

if __name__ == "__main__":
    rgb_p, dep_p, view = sys.argv[1:4]
    sys.path.insert(0, r"D:\weight_detect")
    from goat_detect import detect_goat, segment_box, load_rgb
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    model = joblib.load(os.path.join(MODELDIR, "weight_model_%s.joblib" % view))
    rgb = load_rgb(rgb_p); depth = np.array(Image.open(dep_p)).astype(np.int32)
    box, s, who = detect_goat(rgb)
    if box is None: raise SystemExit("no goat detected")
    mask = segment_box(rgb, box) > 0
    f, err = metric_feats(mask, depth, model["fx"])
    if f is None: raise SystemExit("cannot measure: " + err)
    kg, why = predict_weight(model, f)
    os.makedirs(OUTDIR, exist_ok=True)
    ov = rgb.copy(); ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    txt = ("%.1f kg  (typical error +/- %.0f kg)" % (kg, model["cv_mae"])) if kg else "NO READING: " + "; ".join(why)
    sc = 1400 / ov.shape[1]; ov = cv2.resize(ov, None, fx=sc, fy=sc)
    cv2.putText(ov, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    out = os.path.join(OUTDIR, os.path.basename(os.path.dirname(os.path.dirname(rgb_p))) + "_" + os.path.splitext(os.path.basename(rgb_p))[0] + "_%s_weight.jpg" % view)
    cv2.imwrite(out, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    print(txt); print("features:", {k: round(v, 1) if isinstance(v, float) else v for k, v in f.items()}); print("saved", out)

