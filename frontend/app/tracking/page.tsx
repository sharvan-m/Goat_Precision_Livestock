"use client";

import { useState } from "react";
import { Activity, Loader2, Upload } from "lucide-react";
import { postForm } from "@/lib/api";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
function displayId(value:any){ const s=String(value ?? ""); const m=s.match(/(\d+)$/); return m ? m[1].padStart(2,"0") : s; }

export default function TrackingPage() {
  const [file,setFile]=useState<File|null>(null);
  const [busy,setBusy]=useState(false);
  const [result,setResult]=useState<any>(null);
  const [error,setError]=useState("");

  async function run(){
    if(!file) return;
    setBusy(true);setError("");setResult(null);
    try{
      const f=new FormData();f.append("file",file);f.append("confidence","0.35");f.append("activate","0.25");f.append("lost_buffer","45");f.append("match_threshold","0.80");f.append("minimum_frames","1");
      setResult(await postForm("/api/tracking/video",f));
    }catch(e){setError(e instanceof Error?e.message:"Tracking failed.");}
    finally{setBusy(false);}
  }

  return <>
    <div className="page-top"><div><div className="eyebrow">LIVE IDENTITY</div><h1>Follow every goat through the frame.</h1><p className="page-copy">A dedicated identity workflow for video. ByteTrack maintains goat IDs across frames and produces an annotated video plus tracking CSV.</p></div></div>
    <div className="grid-2">
      <div className="card">
        <div className="dropzone"><Upload size={25} style={{color:"var(--green)"}}/><h3>Upload a goat video</h3><p className="page-copy" style={{margin:"0 auto"}}>MP4, AVI, MOV or MKV</p><input type="file" accept=".mp4,.avi,.mov,.mkv" onChange={e=>setFile(e.target.files?.[0]||null)}/></div>
        {file && <div className="notice" style={{marginTop:12}}>{file.name}</div>}
        <button className="btn btn-primary" onClick={run} disabled={!file||busy} style={{width:"100%",justifyContent:"center",marginTop:12,opacity:(!file||busy)?.55:1}}>
          {busy?<><Loader2 size={17}/> Tracking…</>:<><Activity size={17}/> Start ID tracking</>}
        </button>
        {error && <div className="notice error" style={{marginTop:12}}>{error}</div>}
      </div>
      <div className="card">
        <h2>Tracking logic</h2>
        <div className="feature-card" style={{marginTop:12}}><b>01 · Detect</b><p className="page-copy">Find animal candidates using the existing detector pipeline.</p></div>
        <div className="feature-card" style={{marginTop:10}}><b>02 · Associate</b><p className="page-copy">ByteTrack links detections across frames.</p></div>
        <div className="feature-card" style={{marginTop:10}}><b>03 · Persist</b><p className="page-copy">The same track ID is displayed for the same tracked goat.</p></div>
      </div>
    </div>

    {result && <section style={{marginTop:18}}>
      <div className="stats">
        <div className="stat"><div className="stat-label">Unique goats</div><div className="stat-value">{result.total_unique_goats}</div></div>
        <div className="stat"><div className="stat-label">Frames</div><div className="stat-value">{result.total_frames}</div></div>
        <div className="stat"><div className="stat-label">Detection events</div><div className="stat-value">{result.csv_rows_count}</div></div>
        <div className="stat"><div className="stat-label">Weight</div><div className="stat-value">Separate</div><div className="stat-note">Tracking workflow does not calculate weight.</div></div>
      </div>
      <div className="card"><h2>Annotated output</h2><div className="preview"><video controls src={`${API}${result.output_video_url}`}/></div><a className="btn btn-primary" href={`${API}${result.output_video_url}`} style={{marginTop:12}}>Open annotated video</a></div>
      <div className="section-head"><div><h2>Per-goat identity summary</h2></div><a className="btn btn-ghost" href={`${API}${result.output_csv_url}`}>Download tracking CSV</a></div>
      <div className="table-wrap"><table><thead><tr><th>Goat ID</th><th>Summary</th></tr></thead><tbody>{(result.goat_summary||[]).map((g:any,i:number)=><tr key={i}><td><b>{displayId(g.tracker_id ?? g.id ?? i+1)}</b></td><td>{JSON.stringify(g)}</td></tr>)}</tbody></table></div>
    </section>}
  </>;
}
