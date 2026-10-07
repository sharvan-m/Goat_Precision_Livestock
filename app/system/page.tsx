"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, CloudCog, Database, RefreshCw, Server, Workflow } from "lucide-react";
import { getJSON } from "@/lib/api";

export default function SystemPage() {
  const [health,setHealth]=useState<any|null>(null);
  const [model,setModel]=useState<any|null>(null);
  const [rf,setRf]=useState<any|null>(null);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState("");

  const refresh=useCallback(async()=>{
    setLoading(true);setError("");
    try{
      const [h,m,r]=await Promise.all([
        getJSON<any>("/api/health"),
        getJSON<any>("/api/model"),
        getJSON<any>("/api/roboflow/status"),
      ]);
      setHealth(h);setModel(m);setRf(r);
    }catch(e){setError(e instanceof Error?e.message:"Unable to reach backend.");}
    finally{setLoading(false);}
  },[]);

  useEffect(()=>{refresh()},[refresh]);

  return <>
    <div className="page-top">
      <div><div className="eyebrow">SYSTEM</div><h1>The engine behind the monitor.</h1><p className="page-copy">One place for backend health, model status and the Roboflow detection connection.</p></div>
      <button className="btn btn-ghost" onClick={refresh}><RefreshCw size={15} className={loading?"spin":""}/> Refresh</button>
    </div>

    {error && <div className="notice warn" style={{marginBottom:14}}><CircleAlert size={14} style={{verticalAlign:"-2px",marginRight:5}}/>{error}</div>}

    <div className="stats">
      <div className="stat"><div className="stat-label">FastAPI</div><div className="stat-value">{health?.status||"—"}</div><div className="stat-note">Python backend</div></div>
      <div className="stat"><div className="stat-label">Compute</div><div className="stat-value">{health?.compute||"—"}</div><div className="stat-note">Current PyTorch device</div></div>
      <div className="stat"><div className="stat-label">Roboflow</div><div className="stat-value">{rf?.configured?"Ready":"Not configured"}</div><div className="stat-note">{rf?.provider||"Workflow provider"}</div></div>
      <div className="stat"><div className="stat-label">Weight model</div><div className="stat-value">{health?.weight_model_loaded?"Loaded":"Fallback"}</div><div className="stat-note">{health?.weight_model_name||"No saved regressor"}</div></div>
    </div>

    <div className="system-grid">
      <div className="card">
        <h2><CloudCog size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Roboflow Workflow</h2>
        <SystemRow label="Connection" value={rf?.configured?"Configured":"Missing API key"} ok={!!rf?.configured}/>
        <SystemRow label="Provider" value={rf?.provider||"Roboflow Workflow"} ok={true}/>
        <SystemRow label="Endpoint" value={rf?.endpoint||"—"} ok={!!rf?.endpoint}/>
        <div className="notice" style={{marginTop:12}}>The API key remains server-side. The browser never receives it.</div>
      </div>
      <div className="card">
        <h2><Workflow size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Active pipeline</h2>
        <SystemRow label="Goat detection" value="Roboflow Workflow" ok={!!rf?.configured}/>
        <SystemRow label="Weight estimation" value={health?.weight_model_loaded?"Python regressor":"Fallback estimator"} ok={true}/>
        <SystemRow label="Tracking" value="ByteTrack / existing tracker" ok={true}/>
        <SystemRow label="BCS" value={health?.bcs_available?"Connected":"Pending"} ok={!!health?.bcs_available}/>
      </div>
    </div>

    <div className="section-head"><div><h2>Architecture</h2><p className="page-copy">The browser stays presentation-focused while Python owns the models and external AI calls.</p></div></div>
    <div className="architecture">
      <Arch num="01" title="Next.js" text="Operator UI, upload, previews, result cards and clean labels."/>
      <Arch num="02" title="FastAPI" text="Authenticates and calls the Roboflow Workflow, then exposes results to the UI."/>
      <Arch num="03" title="Roboflow" text="General Segmentation API 51 receives image + goat class prompt."/>
      <Arch num="04" title="Existing engine" text="Weight estimation, tracking, logs and output files stay in your Python project."/>
    </div>

    <div className="grid-2" style={{marginTop:14}}>
      <div className="card"><h2><Server size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Model status</h2><div className="code-block">{model?.path||"Model path unavailable"}{"\n"}{model?.name||"No model loaded"}</div></div>
      <div className="card"><h2><Database size={17} style={{verticalAlign:"-3px",marginRight:7}}/> Data status</h2><SystemRow label="Dataset" value={model?.dataset_present?"Present":"Missing"} ok={!!model?.dataset_present}/><SystemRow label="CSV log" value={model?.csv_log_present?"Present":"Missing"} ok={!!model?.csv_log_present}/><SystemRow label="Excel log" value={model?.excel_log_present?"Present":"Missing"} ok={!!model?.excel_log_present}/></div>
    </div>
  </>;
}

function SystemRow({label,value,ok}:{label:string,value:string,ok:boolean}) {
  return <div className="system-row"><span className="system-label">{label}</span><span className="system-value">{ok&&<CheckCircle2 size={14} color="var(--green)" style={{verticalAlign:"-3px",marginRight:5}}/>}{value}</span></div>
}
function Arch({num,title,text}:{num:string,title:string,text:string}) {
  return <div className="arch-step"><div className="num">{num}</div><b>{title}</b><p>{text}</p></div>
}
