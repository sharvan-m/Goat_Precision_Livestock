"use client";

import { useState } from "react";
import { AlertCircle, CircleCheck, Loader2, ScanSearch, Sparkles } from "lucide-react";
import { postForm } from "@/lib/api";
import { UploadDropzone } from "@/components/UploadDropzone";

type Prediction = {
  class: string;
  confidence: number | null;
  box: {x1:number;y1:number;x2:number;y2:number} | null;
};

type Response = {
  source_image: string;
  annotated_image: string | null;
  goats_detected: number;
  predictions: Prediction[];
  workspace: string;
  workflow: string;
  image_width: number;
  image_height: number;
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function idFor(i:number){ return String(i+1).padStart(2,"0"); }

export default function GoatDetectionPage() {
  const [file,setFile]=useState<File|null>(null);
  const [result,setResult]=useState<Response|null>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");

  async function scan(){
    if(!file)return;
    setBusy(true);setError("");setResult(null);
    try{
      const form=new FormData();
      form.append("file",file);
      setResult(await postForm<Response>("/api/detect/goats",form));
    }catch(e){
      setError(e instanceof Error?e.message:"Roboflow detection failed.");
    }finally{setBusy(false);}
  }

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">ROBOFLOW · SAM 3 WORKFLOW</div>
        <h1>See every goat in the frame.</h1>
        <p className="page-copy">
          Your deployed Roboflow workflow is used for goat concept segmentation.
          The UI keeps the result clean: small numeric IDs over detected goats and a separate model-annotation view.
        </p>
      </div>
      <div className="status good"><CircleCheck size={14}/> Workflow integration</div>
    </div>

    <div className="card">
      <div className="grid-2">
        <div>
          <UploadDropzone file={file} onFile={setFile}/>
        </div>
        <div>
          <div className="feature-card">
            <div className="feature-icon"><ScanSearch size={17}/></div>
            <h3>Roboflow Goat Workflow</h3>
            <p className="page-copy">
              The backend sends the uploaded image to the configured Roboflow Workflow with
              <b> classes = ["goat"]</b>, then returns predictions and the workflow visualization when available.
            </p>
          </div>
          <button
            className="btn btn-primary"
            disabled={!file || busy}
            style={{width:"100%",marginTop:12,opacity:(!file||busy)?.55:1}}
            onClick={scan}
          >
            {busy ? <><Loader2 size={16} className="spin"/> Detecting goats…</> : <><Sparkles size={16}/> Detect & annotate</>}
          </button>
          {error && <div className="notice error" style={{marginTop:12}}><AlertCircle size={15} style={{verticalAlign:"-3px",marginRight:5}}/>{error}</div>}
        </div>
      </div>
    </div>

    {result && (
      <section style={{marginTop:18}}>
        <div className="stats">
          <div className="stat"><div className="stat-label">Goats detected</div><div className="stat-value">{result.goats_detected}</div><div className="stat-note">Filtered to the configured goat class</div></div>
          <div className="stat"><div className="stat-label">Workflow</div><div className="stat-value" style={{fontSize:18}}>Connected</div><div className="stat-note">{result.workflow}</div></div>
          <div className="stat"><div className="stat-label">Classes</div><div className="stat-value">Goat</div><div className="stat-note">Single-class prompt</div></div>
          <div className="stat"><div className="stat-label">Annotation</div><div className="stat-value">{result.annotated_image ? "Available" : "Clean UI"}</div><div className="stat-note">{result.annotated_image ? "Roboflow visualization returned" : "Frontend labels are used"}</div></div>
        </div>

        <div className="section-head"><div><h2>Detection result</h2><p className="page-copy">IDs are placed above the model's detected boxes without drawing the large legacy labels.</p></div></div>
        <div className="result-layout">
          <div className="preview">
            <div className="image-stage">
              <img src={result.source_image} alt="Goat detection result"/>
              {result.predictions.map((p,i)=>{
                if(!p.box)return null;
                const cx=((p.box.x1+p.box.x2)/2);
                const top=Math.max(2,p.box.y1);
                return (
                  <span
                    key={`${i}-${p.class}`}
                    className="id-chip"
                    style={{
                      left:`${(cx/result.image_width)*100}%`,
                      top:`${(top/result.image_height)*100}%`
                    }}
                  >{idFor(i)}</span>
                )
              })}
            </div>
          </div>
          <div>
            {result.predictions.map((p,i)=>(
              <div className="result-card" key={i}>
                <div className="result-top"><h3>Goat {idFor(i)}</h3><span className="status good">DETECTED</span></div>
                <div className="result-big">{p.confidence == null ? "Confidence —" : `${(p.confidence*100).toFixed(1)}%`}</div>
                <div className="result-grid">
                  <div className="kv"><span>Class</span><b>{p.class || "goat"}</b></div>
                  <div className="kv"><span>Type</span><b>Segmentation</b></div>
                </div>
              </div>
            ))}
            {result.predictions.length===0 && <div className="card empty">No goat predictions were returned by the workflow.</div>}
          </div>
        </div>

        {result.annotated_image && (
          <details className="card" style={{marginTop:14}}>
            <summary style={{fontWeight:800,cursor:"pointer"}}>Show Roboflow model annotation</summary>
            <p className="page-copy" style={{margin:"10px 0"}}>This is the visualization returned by your Roboflow Workflow. The operator-facing view above intentionally stays cleaner.</p>
            <div className="preview"><img src={result.annotated_image} alt="Roboflow workflow annotation"/></div>
          </details>
        )}
      </section>
    )}
  </>;
}

