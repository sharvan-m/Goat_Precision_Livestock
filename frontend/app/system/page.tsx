"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, CloudCog, Database, RefreshCw, Server, ShieldCheck, Workflow } from "lucide-react";
import { getJSON } from "@/lib/api";

export default function SystemPage() {
  const [health,setHealth]=useState<any|null>(null);
  const [model,setModel]=useState<any|null>(null);
  const [rf,setRf]=useState<any|null>(null);
  const [side,setSide]=useState<any|null>(null);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState("");

  const refresh=useCallback(async()=>{
    setLoading(true);setError("");
    try{
      const [h,m,r,s]=await Promise.all([
        getJSON<any>("/api/health"),
        getJSON<any>("/api/model"),
        getJSON<any>("/api/roboflow/status"),
        getJSON<any>("/api/side/status"),
      ]);
      setHealth(h);setModel(m);setRf(r);setSide(s);
    }catch(e){setError(e instanceof Error?e.message:"Unable to reach backend.");}
    finally{setLoading(false);}
  },[]);

  useEffect(()=>{refresh()},[refresh]);

  return <>
    <div className="page-top">
      <div><div className="eyebrow">SYSTEM</div><h1>The engine behind your results.</h1><p className="page-copy">A clean operational view of the computer-vision, side-view and weight-estimation services.</p></div>
      <button className="btn btn-ghost" onClick={refresh}><RefreshCw size={15} className={loading?"spin":""}/> Refresh</button>
    </div>

    {error && <div className="notice warn" style={{marginBottom:14}}><CircleAlert size={14} style={{verticalAlign:"-2px",marginRight:5}}/>{error}</div>}

    <div className="stats">
      <div className="stat"><div className="stat-label">Backend</div><div className="stat-value">{health?.status||"—"}</div><div className="stat-note">FastAPI</div></div>
      <div className="stat"><div className="stat-label">Goat detector</div><div className="stat-value">Video model</div><div className="stat-note">Same GoatDetector used for photos</div></div>
      <div className="stat"><div className="stat-label">Side view</div><div className="stat-value">Active</div><div className="stat-note">{side?.configured?"Roboflow classifier":"Existing view evaluator"}</div></div>
      <div className="stat"><div className="stat-label">Weight model</div><div className="stat-value">{health?.weight_model_loaded?"Loaded":"Fallback"}</div><div className="stat-note">{health?.weight_model_name||"No saved regressor"}</div></div>
    </div>

    <div className="system-grid">
      <div className="card">
        <h2><CloudCog size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Model + Roboflow</h2>
        <SystemRow label="Workflow" value={rf?.configured ? rf?.workflow || "Connected" : "Not configured"} ok={!!rf?.configured}/>
        <SystemRow label="Goat class filter" value="goat" ok={true}/>
        <SystemRow label="Photo annotation" value="Same video model mask" ok={true}/>
        <div className="notice" style={{marginTop:12}}>The model API key remains in the Python backend. It is never sent to the browser.</div>
      </div>

      <div className="card">
        <h2><Workflow size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Side-view classification</h2>
        <SystemRow label="State" value="ACTIVE" ok={true}/>
        <SystemRow label="Current engine" value={side?.configured ? "Roboflow side classifier" : "BestViewEvaluator"} ok={true}/>
        <SystemRow label="Output" value="SIDE / NOT SIDE" ok={true}/>
        <SystemRow label="Weight gate" value="Suitable view required" ok={true}/>
        <div className="notice" style={{marginTop:12}}>A dedicated Roboflow side classifier can override this fallback when its workspace/workflow is configured in <code>backend/.env</code>.</div>
      </div>
    </div>

    <div className="section-head"><div><h2>Runtime</h2></div></div>
    <div className="card">
      <div className="system-grid">
        <div>
          <SystemRow label="Compute" value={health?.compute||"—"} ok={true}/>
          <SystemRow label="FFmpeg" value={health?.ffmpeg_available?"Available":"Not available"} ok={!!health?.ffmpeg_available}/>
        </div>
        <div>
          <SystemRow label="CSV log" value={model?.csv_log_present?"Present":"Missing"} ok={!!model?.csv_log_present}/>
          <SystemRow label="Excel log" value={model?.excel_log_present?"Present":"Missing"} ok={!!model?.excel_log_present}/>
        </div>
      </div>
    </div>

    <div className="section-head"><div><h2>Architecture</h2><p className="page-copy">Three screens, one pipeline.</p></div></div>
    <div className="architecture">
      <Arch num="01" title="Analyze" text="Upload an image or video. Detection, side view and weight processing happen together."/>
      <Arch num="02" title="Python + Roboflow" text="FastAPI calls the goat segmentation Workflow and the existing Python measurement engine."/>
      <Arch num="03" title="Herd records" text="Results are logged once and become part of the persistent herd record."/>
      <Arch num="04" title="System" text="Health and model status stay visible without exposing technical clutter in Analyze."/>
    </div>

    <div className="card" style={{marginTop:14}}>
      <h2><ShieldCheck size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Model note</h2>
      <p className="page-copy">The dashboard reports model outputs. Field-validated weight accuracy still requires calibrated imaging and independently measured ground truth.</p>
    </div>
  </>;
}

function SystemRow({label,value,ok}:{label:string,value:string,ok:boolean}) {
  return <div className="system-row"><span className="system-label">{label}</span><span className="system-value">{ok&&<CheckCircle2 size={14} color="var(--green)" style={{verticalAlign:"-3px",marginRight:5}}/>}{value}</span></div>
}
function Arch({num,title,text}:{num:string,title:string,text:string}) {
  return <div className="arch-step"><div className="num">{num}</div><b>{title}</b><p>{text}</p></div>
}
