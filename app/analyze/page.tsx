"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Camera, CircleCheck, FileVideo, Loader2, Sparkles } from "lucide-react";
import { postForm, mediaUrl } from "@/lib/api";
import { UploadDropzone } from "@/components/UploadDropzone";
import type { AnalysisResult } from "@/lib/types";

type ImageResponse = {
  source_image: string;
  annotated_image?: string | null;
  goats_detected: number;
  width: number;
  height: number;
  results: AnalysisResult[];
  model: string;
};

type VideoJob = {
  job_id: string;
  status: string;
  progress?: number;
  current_frame?: number;
  total_frames?: number;
  fps?: number;
  goats_tracked?: number;
  error?: string;
  result?: {
    total_goats: number;
    output_video_url: string;
    excel_url: string;
    csv_url: string;
    summary: Array<{
      goat_id: string;
      final_weight_kg: number;
      best_view_score: number;
      view_category: string;
      confidence: string;
      valid_frames_count: number;
      best_frame_url: string;
    }>;
  };
};

function displayId(id: string) {
  const m = id.match(/(\d+)$/);
  return m ? m[1].padStart(2, "0") : id;
}

export default function AnalyzePage() {
  const [mode, setMode] = useState<"image" | "video">("image");
  const [file, setFile] = useState<File | null>(null);
  const [scale, setScale] = useState("1");
  const [stride, setStride] = useState("2");
  const [busy, setBusy] = useState(false);
  const [imageResult, setImageResult] = useState<ImageResponse | null>(null);
  const [job, setJob] = useState<VideoJob | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setImageResult(null); setJob(null); setError(""); setFile(null);
  }, [mode]);

  async function runImage() {
    if (!file) return;
    setBusy(true); setError(""); setImageResult(null);
    try {
      const form = new FormData();
      form.append("file", file); form.append("pixel_scale", scale);
      setImageResult(await postForm<ImageResponse>("/api/analyze/image", form));
    } catch (e) { setError(e instanceof Error ? e.message : "Image analysis failed."); }
    finally { setBusy(false); }
  }

  async function runVideo() {
    if (!file) return;
    setBusy(true); setError(""); setJob(null);
    try {
      const form = new FormData();
      form.append("file", file); form.append("frame_stride", stride); form.append("pixel_scale", scale);
      const started = await postForm<{ job_id: string }>("/api/analyze/video", form);
      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 850));
        const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/jobs/${started.job_id}`, { cache: "no-store" });
        if (!r.ok) throw new Error(await r.text());
        const next = await r.json(); setJob(next);
        done = next.status === "completed" || next.status === "failed";
      }
    } catch (e) { setError(e instanceof Error ? e.message : "Video processing failed."); }
    finally { setBusy(false); }
  }

  return <>
    <div className="page-top"><div><div className="eyebrow">ANALYZE</div><h1>Turn footage into clean goat records.</h1><p className="page-copy">Upload first. Then see one clean output image, followed by four result cards per row.</p></div></div>

    <div className="card analysis-input-card">
      <div className="analysis-mode-row">
        <div className="segmented-control">
          <button className={mode === "image" ? "active" : ""} onClick={() => setMode("image")}><Camera size={16}/> Image</button>
          <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")}><FileVideo size={16}/> Video</button>
        </div>
        <span className="input-helper">Supported {mode === "image" ? "image" : "video"} upload</span>
      </div>
      <div className="analysis-input-grid">
        <UploadDropzone file={file} onFile={setFile}/>
        <div className="settings-panel">
          <div className="form-row">
            <div className="field"><label>Calibration scale (cm / pixel)</label><input value={scale} onChange={e=>setScale(e.target.value)} type="number" min="0.01" step="0.1"/></div>
            {mode === "video" && <div className="field"><label>Frame stride</label><input value={stride} onChange={e=>setStride(e.target.value)} type="number" min="1" max="5"/></div>}
          </div>
          <div className="notice"><b>Operator flow</b><br/>Detect → validate view → estimate when suitable → save record.</div>
          <button disabled={!file || busy} className="btn btn-primary analyze-button" onClick={mode === "image" ? runImage : runVideo}>
            {busy ? <><Loader2 size={16} className="spin"/> Processing…</> : <><Sparkles size={16}/> Run {mode} analysis</>}
          </button>
          {error && <div className="notice error" style={{marginTop:10}}><AlertCircle size={15} style={{verticalAlign:"-3px",marginRight:5}}/>{error}</div>}
        </div>
      </div>
    </div>

    {imageResult && <section className="analysis-output">
      <div className="section-head"><div><div className="eyebrow">OUTPUT</div><h2>Annotated detection</h2><p className="page-copy">{imageResult.goats_detected} goat{imageResult.goats_detected === 1 ? "" : "s"} detected · {imageResult.model}</p></div><div className="status good"><CircleCheck size={14}/> Annotation ready</div></div>
      <div className="output-image-card">
        <div className="output-image-wrap">
          <img src={imageResult.annotated_image || imageResult.source_image} alt="Annotated goat detection result" className="output-image"/>
        </div>
      </div>
      <div className="section-head result-section-head"><div><h2>Goat results</h2><p className="page-copy">Weight, BCS state, view quality and detector confidence for each detected goat.</p></div></div>
      {imageResult.results.length === 0 ? <div className="card empty">No goat detections were returned.</div> : <div className="goat-result-grid">{imageResult.results.map(r => <ResultCard key={r.goat_id} r={r}/>)}</div>}
    </section>}

    {job && <section className="analysis-output">
      <div className="section-head"><div><div className="eyebrow">VIDEO OUTPUT</div><h2>Processing result</h2></div></div>
      <div className="card"><div className="progress-header"><b>{job.status === "completed" ? "Processing complete" : job.status === "failed" ? "Processing failed" : "Processing…"}</b><span>{Math.round((job.progress||0)*100)}%</span></div><div className="progress"><div style={{width:`${Math.round((job.progress||0)*100)}%`}}/></div><p className="muted" style={{fontSize:11}}>Frame {job.current_frame||0} / {job.total_frames||0} · {job.fps||0} FPS · {job.goats_tracked||0} goats tracked</p>{job.error && <div className="notice error">{job.error}</div>}</div>
      {job.result && <><div className="stats"><div className="stat"><div className="stat-label">Unique goats</div><div className="stat-value">{job.result.total_goats}</div></div><div className="stat"><div className="stat-label">Tracking</div><div className="stat-value">Stable IDs</div></div><div className="stat"><div className="stat-label">BCS</div><div className="stat-value">Pending</div></div><div className="stat"><div className="stat-label">Exports</div><div className="stat-value">Ready</div></div></div><div className="output-image-card"><div className="video-wrap"><video controls src={mediaUrl(job.result.output_video_url)}/></div></div><div className="section-head"><div><h2>Final goat summary</h2></div><div style={{display:"flex",gap:8}}><a className="btn btn-ghost" href={mediaUrl(job.result.csv_url)}>Download CSV</a><a className="btn btn-primary" href={mediaUrl(job.result.excel_url)}>Download Excel</a></div></div><div className="table-wrap"><table><thead><tr><th>ID</th><th>Weight</th><th>View</th><th>Score</th><th>Confidence</th><th>Valid frames</th></tr></thead><tbody>{job.result.summary.map(s=><tr key={s.goat_id}><td><b>{displayId(s.goat_id)}</b></td><td>{Number(s.final_weight_kg).toFixed(1)} kg</td><td>{s.view_category}</td><td>{Number(s.best_view_score).toFixed(0)}/100</td><td>{s.confidence}</td><td>{s.valid_frames_count}</td></tr>)}</tbody></table></div></>}
    </section>}
  </>;
}

function ResultCard({r}:{r:AnalysisResult}) {
  const cls = r.view_category === "GOOD" ? "good" : r.view_category === "ACCEPTABLE" ? "acceptable" : "poor";
  return <div className="result-card">
    <div className="result-top"><div className="goat-name"><span className="goat-number">{displayId(r.goat_id)}</span><h3>Goat {displayId(r.goat_id)}</h3></div><span className={`status ${cls}`}>{r.view_category} · {r.view_score}/100</span></div>
    <div className="result-big">{r.weight_kg != null ? `${r.weight_kg.toFixed(1)} kg` : "Weight unavailable"}</div>
    <div className="muted result-subline">{r.weight_kg != null ? `Confidence: ${r.confidence || "—"}` : (r.rejection_reasons.join(", ") || "Waiting for a suitable view.")}</div>
    <div className="result-grid"><div className="kv"><span>Detector</span><b>{r.detector_confidence}%</b></div><div className="kv"><span>View</span><b>{r.view_category}</b></div><div className="kv"><span>BCS</span><b>Unavailable</b></div><div className="kv"><span>Estimation</span><b>{r.can_estimate_weight ? "Allowed" : "Blocked"}</b></div></div>
    {r.measurements && <details className="measure-details"><summary>Morphometric features</summary><pre className="code-block">{JSON.stringify(r.measurements,null,2)}</pre></details>}
  </div>;
}
