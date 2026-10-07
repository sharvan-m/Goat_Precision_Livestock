"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, HeartPulse, Info, Loader2, RefreshCw, UploadCloud, X } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Model = {
  available:boolean; name?:string; task?:string; path?:string|null; reason?:string;
  model_root?:string; model_count?:number; metadata_count?:number; models_found?:string[];
};
type Prediction = {
  id:number; class_name:string; confidence:number; famacha_score:number|null;
};
type Result = {
  status:string; model:Model; predictions:Prediction[];
  famacha_score:number|null; risk_level:string; annotated_image:string|null; message:string;
};

export default function AnemiaPage(){
  const [file,setFile]=useState<File|null>(null);
  const [preview,setPreview]=useState("");
  const [model,setModel]=useState<Model|null>(null);
  const [result,setResult]=useState<Result|null>(null);
  const [busy,setBusy]=useState(false);
  const [drag,setDrag]=useState(false);
  const input=useRef<HTMLInputElement>(null);

  const refresh=useCallback(async()=>{
    try{
      const r=await fetch(`${API}/api/anemia/status`,{cache:"no-store"});
      setModel(await r.json());
    }catch{ setModel(null); }
  },[]);

  useEffect(()=>{refresh()},[refresh]);
  useEffect(()=>{
    if(!file){setPreview("");return}
    const u=URL.createObjectURL(file); setPreview(u);
    return()=>URL.revokeObjectURL(u);
  },[file]);

  const pick=(files:FileList|null)=>{
    if(files?.length){setFile(files[0]);setResult(null);}
  };

  async function analyze(){
    if(!file)return;
    setBusy(true);setResult(null);
    try{
      const f=new FormData();f.append("file",file);
      const r=await fetch(`${API}/api/anemia/analyze`,{method:"POST",body:f});
      const p=await r.json();
      if(!r.ok)throw new Error(p.detail||"Anemia analysis failed.");
      setResult(p);
      await refresh();
    }catch(e){
      setResult({
        status:"error",model:model||({} as Model),predictions:[],
        famacha_score:null,risk_level:"Cannot determine",
        annotated_image:null,
        message:e instanceof Error?e.message:"Anemia analysis failed.",
      });
    }finally{setBusy(false);}
  }

  const score=result?.famacha_score??null;
  const tone=score===null?"neutral":score>=4?"poor":score===3?"acceptable":"good";
  const scoreText=score ? `${score} · ${score===1?"Red":score===2?"Red-pink":score===3?"Pink":score===4?"Pale pink":"White"}` : "Not assessable";

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">HEALTH SCREENING · FAMACHA</div>
        <h1>Goat anemia screening.</h1>
        <p className="page-copy">Upload an eye image and run the trained model from the existing anemia project. The weight-analysis pipeline remains separate.</p>
      </div>
      <button className="btn btn-ghost" onClick={refresh}><RefreshCw size={15}/> Model status</button>
    </div>

    <div className="grid-2">
      <div className="card">
        <h2><HeartPulse size={18} style={{verticalAlign:"-3px",marginRight:7}}/> Upload eye image</h2>
        <p className="page-copy" style={{marginBottom:14}}>Use a close, focused view with the lower eyelid/conjunctiva visible.</p>

        <div
          className={`dropzone ${drag?"is-dragging":""}`}
          onDragEnter={e=>{e.preventDefault();setDrag(true)}}
          onDragOver={e=>e.preventDefault()}
          onDragLeave={e=>{e.preventDefault();setDrag(false)}}
          onDrop={e=>{e.preventDefault();setDrag(false);pick(e.dataTransfer.files)}}
          onClick={()=>input.current?.click()}
        >
          <UploadCloud size={30} color="var(--green)"/>
          <div className="dropzone-title">{drag?"Release to upload":"Drop the goat-eye image here"}</div>
          <div className="dropzone-copy">or click to browse · JPG, PNG, BMP, WEBP</div>
          {file&&<div className="dropzone-file" onClick={e=>e.stopPropagation()}>
            <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{file.name}</span>
            <CheckCircle2 size={14} color="var(--green)"/>
            <button onClick={()=>setFile(null)} style={{border:0,background:"transparent",display:"grid",padding:0}}><X size={14}/></button>
          </div>}
          <input ref={input} hidden type="file" accept=".jpg,.jpeg,.png,.bmp,.webp" onChange={e=>pick(e.target.files)}/>
        </div>

        {preview&&<img src={preview} className="anemia-preview" alt="Uploaded goat eye"/>}

        <button className="btn btn-primary" disabled={!file||busy} onClick={analyze} style={{width:"100%",marginTop:13,opacity:!file||busy?.55:1}}>
          {busy?<><Loader2 size={16} className="spin"/> Screening…</>:<><HeartPulse size={16}/> Analyze FAMACHA</>}
        </button>

        <div className="notice" style={{marginTop:12}}>
          <Info size={14} style={{verticalAlign:"-2px",marginRight:5}}/> AI screening aid only; it is not a veterinary diagnosis.
        </div>
      </div>

      <div className="card card-soft">
        <h2>Model discovered</h2>
        <div className={`model-ready ${model?.available?"ready":"missing"}`}>
          {model?.available?<CheckCircle2 size={20}/>:<AlertCircle size={20}/>}
          <div><b>{model?.available?model.name:"No anemia model loaded"}</b><p>{model?.available?`${model.task||"YOLO"} · selected from project folder`:model?.reason||"Checking…"}</p></div>
        </div>
        <div className="system-row"><span className="system-label">Folder</span><span className="system-value">{model?.model_root||"D:\\weight_detect\\Anemia\\goat_eye_detection"}</span></div>
        <div className="system-row"><span className="system-label">Models found</span><span className="system-value">{model?.model_count??"—"}</span></div>
        <div className="system-row"><span className="system-label">Metadata found</span><span className="system-value">{model?.metadata_count??"—"}</span></div>
        {(model as any)?.candidates?.length ? <details style={{marginTop:10}}>
          <summary style={{fontSize:11,fontWeight:800,cursor:"pointer"}}>Show model audit</summary>
          <div className="model-audit">
            {(model as any).candidates.map((c:any,i:number)=>(
              <div className="audit-row" key={i}>
                <div>
                  <b>{c.name}</b>
                  <span>{c.task || "unknown"} · {Object.values(c.classes || {}).join(", ") || "no class names"}</span>
                </div>
                <strong className={c.suitable ? "audit-good" : "audit-bad"}>
                  {c.suitable ? "ELIGIBLE" : "EXCLUDED"}
                </strong>
              </div>
            ))}
          </div>
        </details> : null}
      </div>
    </div>

    {result&&<section style={{marginTop:18}}>
      <div className="section-head"><div><div className="eyebrow">SCREENING RESULT</div><h2>FAMACHA / anemia result</h2></div><span className={`status ${tone}`}>{result.status==="ok"?"COMPLETE":"CHECK RESULT"}</span></div>

      <div className="result-layout">
        <div className="preview" style={{minHeight:380}}>
          <img src={result.annotated_image||preview} alt="Anemia model result"/>
        </div>
        <div>
          <div className="result-card">
            <div className="result-top"><h3>FAMACHA score</h3><span className={`status ${tone}`}>{scoreText}</span></div>
            <div className="result-big">{score??"—"}</div>
            <div className="muted" style={{fontSize:11}}>{score?result.risk_level:"The selected eye model did not expose a direct 1–5 FAMACHA class."}</div>
            <div className="famacha-scale">
              {[1,2,3,4,5].map(n=><div key={n} className={`famacha-step ${score===n?"active":""}`}><b>{n}</b><span>{n===1?"Red":n===2?"Red-pink":n===3?"Pink":n===4?"Pale pink":"White"}</span></div>)}
            </div>
          </div>

          <div className="card" style={{marginTop:10}}>
            <h2>Model predictions</h2>
            {result.predictions.length?result.predictions.map((p,i)=><div className="prediction-row" key={i}><div><b>{p.class_name}</b><span>{p.famacha_score?`FAMACHA ${p.famacha_score}`:"No direct FAMACHA mapping"}</span></div><strong>{p.confidence.toFixed(1)}%</strong></div>):<div className="empty" style={{padding:"24px 4px"}}>No predictions returned.</div>}
          </div>

          <div className="notice warn" style={{marginTop:10}}>{result.message}</div>
        </div>
      </div>
    </section>}
  </>;
}
