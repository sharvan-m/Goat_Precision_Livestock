import sys, os
import numpy as np, cv2, torch
from PIL import Image
from ultralytics import SAM

DEV = 0 if torch.cuda.is_available() else "cpu"
_state = {}

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def detect_yoloworld(rgb, conf=0.15):
    from ultralytics import YOLOWorld
    if "yw" not in _state:
        m = YOLOWorld("yolov8s-worldv2.pt")
        m.set_classes(["goat"])
        _state["yw"] = m
    r = _state["yw"].predict(rgb[..., ::-1], conf=conf, device=DEV, verbose=False)[0]
    if len(r.boxes) == 0: return None
    i = int(r.boxes.conf.argmax())
    return r.boxes.xyxy[i].cpu().numpy(), float(r.boxes.conf[i])

def detect_gdino(rgb, thr=0.3):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if "gd" not in _state:
        n = "IDEA-Research/grounding-dino-tiny"
        _state["gd"] = (AutoProcessor.from_pretrained(n),
                        AutoModelForZeroShotObjectDetection.from_pretrained(n).to("cuda" if DEV == 0 else "cpu"))
    proc, model = _state["gd"]
    pil = Image.fromarray(rgb)
    inp = proc(images=pil, text="a goat.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp)
    try:
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=thr, text_threshold=0.25, target_sizes=[pil.size[::-1]])[0]
    if len(res["boxes"]) == 0: return None
    i = int(res["scores"].argmax())
    return res["boxes"][i].cpu().numpy(), float(res["scores"][i])

def detect_goat(rgb):
    for name, fn in (("yolo-world", detect_yoloworld), ("grounding-dino", detect_gdino)):
        try:
            d = fn(rgb)
        except Exception as e:
            print(f"  {name} failed: {e}"); d = None
        if d is not None:
            return d[0], d[1], name
    return None, 0.0, None

def segment_box(rgb, box):
    if "sam" not in _state:
        _state["sam"] = SAM("mobile_sam.pt")
    r = _state["sam"](rgb[..., ::-1], bboxes=[[float(v) for v in box]], device=DEV, verbose=False)[0]
    return r.masks.data[0].cpu().numpy().astype(np.uint8)

def quality(mask, box, shape):
    h, w = shape[:2]
    area = mask.mean()
    ys, xs = np.where(mask)
    touches = xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3 or ys.max() >= h - 3
    ok = 0.02 < area < 0.85 and not touches
    return ok, area, touches

if __name__ == "__main__":
    out = r"D:\weight_detect\det_test"; os.makedirs(out, exist_ok=True)
    for p in sys.argv[1:]:
        rgb = load_rgb(p); print(os.path.basename(p), rgb.shape)
        box, s, who = detect_goat(rgb)
        if box is None:
            print("  NO DETECTION"); continue
        mask = segment_box(rgb, box)
        ok, area, touch = quality(mask, box, rgb.shape)
        print(f"  {who} score={s:.2f} box={box.astype(int).tolist()} mask_area={area:.2f} touches_border={touch} quality_ok={ok}")
        ov = rgb.copy(); ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        x1, y1, x2, y2 = box.astype(int); cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), max(2, rgb.shape[1] // 400))
        sc = 1200 / max(ov.shape[:2]); ov = cv2.resize(ov, None, fx=min(sc, 1), fy=min(sc, 1))
        cv2.imwrite(os.path.join(out, "det_" + os.path.splitext(os.path.basename(p))[0] + ".jpg"), ov[..., ::-1])
