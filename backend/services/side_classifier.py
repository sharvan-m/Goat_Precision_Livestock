"""Optional separate Roboflow side-view classifier."""
from __future__ import annotations
import base64, os
from typing import Any
import requests
API_URL=os.getenv("ROBOFLOW_API_URL","https://serverless.roboflow.com").rstrip("/")
API_KEY=os.getenv("ROBOFLOW_API_KEY","").strip()
WORKSPACE=os.getenv("ROBOFLOW_SIDE_WORKSPACE","").strip()
WORKFLOW=os.getenv("ROBOFLOW_SIDE_WORKFLOW","").strip()
def configured()->bool: return bool(API_KEY and WORKSPACE and WORKFLOW)
def endpoint()->str: return f"{API_URL}/{WORKSPACE}/workflows/{WORKFLOW}"
def _walk(v:Any):
    if isinstance(v,dict):
        for k,x in v.items():
            yield str(k),x; yield from _walk(x)
    elif isinstance(v,list):
        for x in v: yield from _walk(x)
def classify(image_bytes:bytes,timeout:int=90)->dict[str,Any]:
    if not configured(): return {"configured":False,"class":None,"confidence":None}
    payload={"api_key":API_KEY,"inputs":{"image":{"type":"base64","value":base64.b64encode(image_bytes).decode("ascii")}}}
    try:
        r=requests.post(endpoint(),json=payload,headers={"Content-Type":"application/json"},timeout=timeout); r.raise_for_status(); body=r.json()
    except Exception as exc:
        return {"configured":True,"class":None,"confidence":None,"error":str(exc)}
    cls=None; conf=None
    for k,v in _walk(body):
        kl=k.lower()
        if cls is None and kl in {"class","label","predicted_class","top_class"} and isinstance(v,str): cls=v
        if conf is None and kl in {"confidence","score","probability","top_confidence"} and isinstance(v,(int,float)): conf=float(v)
    return {"configured":True,"class":cls,"confidence":conf,"raw":body}
