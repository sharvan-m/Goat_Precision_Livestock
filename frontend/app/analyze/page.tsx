"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Camera, CheckCircle2, Download, FileVideo, Loader2, Sparkles } from "lucide-react";
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
  annotation_source?: string;
  roboflow_error?: string | null;
};

type GoatVideoResult = {
  goat_id: string;
  final_weight_kg: number;
  final_bcs_score?: number | null;
  bcs_label?: string | null;
  best_view_score: number;
  view_category: string;
  side_class?: string | null;
  side_confidence?: number | null;
  confidence: string;
  valid_frames_count: number;
  best_frame_url: string;
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
    download_video_url?: string;
    video_browser_compatible?: boolean;
    excel_url: string;
    csv_url: string;
    summary: GoatVideoResult[];
  };
};

function displayId(id: string) {
  const m = id.match(/(\d+)$/);
  return m ? m[1].padStart(2, "0") : id;
}
function statusClass(value: string) {
  return value === "GOOD" ? "good" : value === "ACCEPTABLE" ? "acceptable" : value === "POOR" ? "poor" : "neutral";
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
  const [videoError, setVideoError] = useState(false);

  useEffect(() => {
    setFile(null); setImageResult(null); setJob(null); setError(""); setVideoError(false);
  }, [mode]);

  async function runImage() {
    if (!file) return;
    setBusy(true); setError(""); setImageResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("pixel_scale", scale);
      setImageResult(await postForm<ImageResponse>("/api/analyze/image", form));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Image analysis failed.");
    } finally { setBusy(false); }
  }

  async function runVideo() {
    if (!file) return;
    setBusy(true); setError(""); setJob(null); setVideoError(false);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("frame_stride", stride);
      form.append("pixel_scale", scale);
      const started = await postForm<{job_id:string}>("/api/analyze/video", form);
      let done = false;
      while (!done) {
        await new Promise(r => setTimeout(r, 850));
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/jobs/${started.job_id}`, { cache:"no-store" });
        if (!response.ok) throw new Error(await response.text());
        const next = await response.json();
        setJob(next);
        done = next.status === "completed" || next.status === "failed";
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Video processing failed.");
    } finally { setBusy(false); }
  }

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">ANALYZE</div>
        <h1>Detect, validate and measure.</h1>
        <p className="page-copy">One workflow for the operator: The same GoatDetector used by video analysis detects the goats in photos too; side-view validation then gates the weight estimate.</p>
      </div>
    </div>

    <div className="card analysis-input-card">
      <div className="analysis-mode-row">
        <div className="segmented-control">
          <button className={mode === "image" ? "active" : ""} onClick={() => setMode("image")}><Camera size={16}/> Image</button>
          <button className={mode === "video" ? "active" : ""} onClick={() => setMode("video")}><FileVideo size={16}/> Video</button>
        </div>
        <span className="input-helper">Same video detector · side-view classification · weight estimation</span>
      </div>

      <div className="analysis-input-grid">
        <UploadDropzone
          file={file}
          onFile={setFile}
          accept={mode === "video" ? ".mp4,.avi,.mov,.mkv,video/mp4,video/x-msvideo,video/quicktime,video/x-matroska" : ".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp"}
          helperText={mode === "video" ? "or click to browse · MP4, AVI, MOV or MKV" : "or click to browse · JPG, PNG, BMP or WEBP"}
        />
        <div className="settings-panel">
          <div className="form-row">
            <div className="field"><label>Calibration scale (cm / pixel)</label><input value={scale} onChange={e=>setScale(e.target.value)} type="number" min="0.01" step="0.1"/></div>
            {mode === "video" ? <div className="field"><label>Frame stride</label><input value={stride} onChange={e=>setStride(e.target.value)} type="number" min="1" max="5"/></div> : null}
          </div>
          <div className="analysis-mini-grid">
            <div className="mini-status"><span>Goat detection</span><b>Same as video</b></div>
            <div className="mini-status"><span>Side view</span><b>Active</b></div>
            <div className="mini-status"><span>Weight</span><b>Model ready</b></div>
          </div>
          <div className="notice"><b>View gate:</b> a goat can be detected from any angle, but weight estimation is gated by the side/view evaluation.</div>
          <button disabled={!file || busy} className="btn btn-primary analyze-button" onClick={mode === "image" ? runImage : runVideo}>
            {busy ? <><Loader2 size={16} className="spin"/> Processing…</> : <><Sparkles size={16}/> Run {mode} analysis</>}
          </button>
          {error && <div className="notice error" style={{marginTop:10}}><AlertCircle size={15} style={{verticalAlign:"-3px",marginRight:5}}/>{error}</div>}
        </div>
      </div>
    </div>

    {imageResult && <ImageOutput result={imageResult}/>} 
    {job && <VideoOutput job={job} videoError={videoError} setVideoError={setVideoError}/>} 
  </>;
}

function ImageOutput({result}:{result:ImageResponse}) {
  return <section className="analysis-output">
    <div className="section-head">
      <div><div className="eyebrow">OUTPUT</div><h2>Model annotation</h2><p className="page-copy">{result.goats_detected} goat{result.goats_detected === 1 ? "" : "s"} detected · {result.annotation_source || "Annotation ready"}</p></div>
      <div className="status good"><CheckCircle2 size={14}/> Model result ready</div>
    </div>

    <div className="output-image-card">
      <div className="output-image-wrap">
        <img src={result.annotated_image || result.source_image} alt="Goat segmentation annotation" className="output-image"/>
      </div>
    </div>

    {result.roboflow_error && <div className="notice warn" style={{marginTop:10}}>Cloud vision service was unavailable; local detector was used as fallback.</div>}

    <div className="section-head result-section-head"><div><h2>Goat results</h2><p className="page-copy">Each result includes side-view status before weight estimation.</p></div></div>
    {result.results.length === 0 ? <div className="card empty">No goat detections were returned.</div> : <div className="goat-result-grid">{result.results.map(r=><ResultCard key={r.goat_id} r={r}/>)}</div>}
  </section>;
}

function ResultCard({r}:{r:AnalysisResult}) {
  const cls = statusClass(r.view_category);
  const bcsScore = (r as any).bcs_score as number | null | undefined;
  const bcsLabel = (r as any).bcs_label as string | null | undefined;
  const marginKg = (r as any).weight_margin_kg as number | null | undefined;
  return <div className="result-card">
    <div className="result-top"><div className="goat-name"><span className="goat-number">{displayId(r.goat_id)}</span><h3>Goat {displayId(r.goat_id)}</h3></div><span className={`status ${cls}`}>{r.view_category} {r.view_score != null ? ` · ${r.view_score}/100` : ""}</span></div>
    <div className="result-big">{r.weight_kg != null ? `${r.weight_kg.toFixed(1)} kg` : "Weight unavailable"}{r.weight_kg != null && marginKg != null ? <span className="result-margin"> ± {marginKg} kg</span> : null}</div>
    {bcsScore != null && <div className="bcs-row"><span className="bcs-badge">BCS {bcsScore.toFixed(1)}</span><span className="muted bcs-label">{bcsLabel}</span></div>}
    <div className="muted result-subline">{r.weight_kg != null ? `Confidence: ${r.confidence || "—"}` : (r.rejection_reasons.join(", ") || "Waiting for a suitable side view.")}</div>
    <div className="result-grid">
      <div className="kv"><span>Detector</span><b>{r.detector_confidence == null ? "—" : `${r.detector_confidence}%`}</b></div>
      <div className="kv"><span>Side view</span><b>{r.side_class || r.view_category || "—"}</b></div>
      <div className="kv"><span>Side confidence</span><b>{r.side_confidence != null ? `${r.side_confidence}%` : "—"}</b></div>
      <div className="kv"><span>Estimation</span><b>{r.can_estimate_weight ? "Allowed" : "Blocked"}</b></div>
    </div>
    {r.measurements && <details className="measure-details"><summary>Morphometric features</summary><pre className="code-block">{JSON.stringify(r.measurements,null,2)}</pre></details>}
  </div>;
}

function VideoOutput({job,videoError,setVideoError}:{job:VideoJob,videoError:boolean,setVideoError:(v:boolean)=>void}) {
  return <section className="analysis-output">
    <div className="section-head"><div><div className="eyebrow">VIDEO OUTPUT</div><h2>Processing result</h2><p className="page-copy">The video workflow keeps stable IDs across frames and aggregates measurements by goat.</p></div>{job.status === "completed" && <div className="status good"><CheckCircle2 size={14}/> Output ready</div>}</div>

    <div className="card">
      <div className="progress-header"><b>{job.status === "completed" ? "Processing complete" : job.status === "failed" ? "Processing failed" : "Processing…"}</b><span>{Math.round((job.progress||0)*100)}%</span></div>
      <div className="progress"><div style={{width:`${Math.round((job.progress||0)*100)}%`}}/></div>
      <p className="muted" style={{fontSize:11}}>Frame {job.current_frame||0} / {job.total_frames||0} · {job.fps||0} FPS · {job.goats_tracked||0} goats tracked</p>
      {job.error && <div className="notice error">{job.error}</div>}
    </div>

    {job.result && <>
      <div className="stats">
        <div className="stat"><div className="stat-label">Unique goats</div><div className="stat-value">{job.result.total_goats}</div></div>
        <div className="stat"><div className="stat-label">Identity</div><div className="stat-value">Stable IDs</div></div>
        <div className="stat"><div className="stat-label">Side view</div><div className="stat-value">Active</div></div>
        <div className="stat"><div className="stat-label">Exports</div><div className="stat-value">Ready</div></div>
      </div>

      <div className="output-image-card">
        <div className="video-wrap">
          {!videoError ? <video controls playsInline preload="metadata" src={mediaUrl(job.result.output_video_url)} poster={job.result.summary[0]?mediaUrl(job.result.summary[0].best_frame_url):undefined} onError={()=>setVideoError(true)}/> : <div className="empty"><h3>Browser playback could not decode the generated video.</h3><p className="page-copy" style={{margin:"6px auto 14px"}}>The processed file is still available for download.</p><a className="btn btn-primary" href={mediaUrl(job.result.download_video_url || job.result.output_video_url)}><Download size={15}/> Download annotated video</a></div>}
        </div>
      </div>

      {!videoError && <div style={{display:"flex",justifyContent:"flex-end",marginTop:10}}><a className="btn btn-ghost" href={mediaUrl(job.result.download_video_url || job.result.output_video_url)}><Download size={15}/> Download video</a></div>}

      <div className="section-head result-section-head"><div><h2>Goat results</h2><p className="page-copy">One result card per tracked goat.</p></div><div style={{display:"flex",gap:8}}><a className="btn btn-ghost" href={mediaUrl(job.result.csv_url)}>Download CSV</a><a className="btn btn-primary" href={mediaUrl(job.result.excel_url)}>Download Excel</a></div></div>
      {job.result.summary.length===0 ? <div className="card empty">No goat results returned.</div> : <div className="goat-result-grid">{job.result.summary.map(s=><VideoResultCard key={s.goat_id} s={s}/>)}</div>}
    </>}
  </section>;
}

function VideoResultCard({s}:{s:GoatVideoResult}) {
  const cls = statusClass(s.view_category);
  const side = s.side_class || (s.view_category === "GOOD" || s.view_category === "ACCEPTABLE" ? "SIDE" : "NOT SIDE");
  return <div className="result-card">
    <div className="result-top"><div className="goat-name"><span className="goat-number">{displayId(s.goat_id)}</span><h3>Goat {displayId(s.goat_id)}</h3></div><span className={`status ${cls}`}>{s.view_category} · {Number(s.best_view_score).toFixed(0)}/100</span></div>
    <div className="result-big">{Number(s.final_weight_kg).toFixed(1)} kg</div>
    {s.final_bcs_score != null && <div className="bcs-row"><span className="bcs-badge">BCS {s.final_bcs_score.toFixed(1)}</span><span className="muted bcs-label">{s.bcs_label}</span></div>}
    <div className="muted result-subline">Confidence: {s.confidence} · {s.valid_frames_count} valid frames</div>
    <img src={mediaUrl(s.best_frame_url)} alt={`Best frame ${s.goat_id}`} style={{width:"100%",borderRadius:11,marginTop:10,display:"block"}}/>
    <div className="result-grid"><div className="kv"><span>Side view</span><b>{side}</b></div><div className="kv"><span>Side confidence</span><b>{s.side_confidence != null ? `${s.side_confidence}%` : Number(s.best_view_score).toFixed(0)+"%"}</b></div><div className="kv"><span>View score</span><b>{Number(s.best_view_score).toFixed(0)}/100</b></div><div className="kv"><span>Valid frames</span><b>{s.valid_frames_count}</b></div></div>
  </div>;
}
