"use client";

import { useState } from "react";
import { AlertCircle, CircleCheck, Loader2, ScanSearch, Sparkles } from "lucide-react";
import { postForm } from "@/lib/api";
import { UploadDropzone } from "@/components/UploadDropzone";

type Prediction = {
  class: string;
  confidence: number | null;
  box: { x1: number; y1: number; x2: number; y2: number } | null;
  side_view_score?: number;
  is_valid_side?: boolean;
  side_category?: string;
  rejection_reasons?: string[];
};

type Response = {
  source_image: string;
  annotated_image: string | null;
  goats_detected: number;
  valid_side_views?: number;
  invalid_views?: number;
  predictions: Prediction[];
  workspace: string;
  workflow: string;
  image_width: number;
  image_height: number;
  annotation_source?: string;
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function idFor(i: number) { return String(i + 1).padStart(2, "0"); }

export default function GoatDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Response | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [detector, setDetector] = useState<"roboflow" | "local">("roboflow");
  const [viewMode, setViewMode] = useState<"annotated" | "original">("annotated");

  async function scan() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("detector", detector);
      setResult(await postForm<Response>("/api/detect/goats", form));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Goat detection failed.");
    } finally {
      setBusy(false);
    }
  }

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">SIDE-VIEW ANGLE & WEIGHT FEASIBILITY</div>
        <h1>Goat Detection & Side Classification</h1>
        <p className="page-copy">
          Detects every goat in the frame and evaluates side-angle visibility for precision weight estimation.
          Valid side angles are annotated in <b>GREEN</b>; invalid/frontal/occluded angles in <b>RED</b>.
        </p>
      </div>
      <div className="status good">
        <CircleCheck size={14}/> {detector === "roboflow" ? "Roboflow SAM 3" : "Local YOLO11"}
      </div>
    </div>

    <div className="card">
      <div className="grid-2">
        <div>
          <UploadDropzone file={file} onFile={setFile}/>
        </div>
        <div>
          <div style={{ marginBottom: 14 }}>
            <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 8, color: "var(--fg-muted, #888)" }}>
              DETECTION ENGINE
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className={`btn ${detector === "roboflow" ? "btn-primary" : "btn-secondary"}`}
                style={{ flex: 1, padding: "8px 12px", fontSize: 13 }}
                onClick={() => setDetector("roboflow")}
              >
                Roboflow SAM 3
              </button>
              <button
                type="button"
                className={`btn ${detector === "local" ? "btn-primary" : "btn-secondary"}`}
                style={{ flex: 1, padding: "8px 12px", fontSize: 13 }}
                onClick={() => setDetector("local")}
              >
                Custom Model (Local)
              </button>
            </div>
          </div>

          <div className="feature-card">
            <div className="feature-icon"><ScanSearch size={17}/></div>
            <h3>Side-View Quality Classification</h3>
            <p className="page-copy">
              Color-codes bounding boxes and outlines:
              <br/>
              🟢 <b>GREEN</b>: Flank & torso visible at the correct side angle (Weight estimation eligible).
              <br/>
              🔴 <b>RED</b>: Front-facing, rear-facing, or heavily occluded (Unsuitable for weight).
            </p>
          </div>
          <button
            className="btn btn-primary"
            disabled={!file || busy}
            style={{ width: "100%", marginTop: 12, opacity: (!file || busy) ? 0.55 : 1 }}
            onClick={scan}
          >
            {busy ? <><Loader2 size={16} className="spin"/> Analyzing angles…</> : <><Sparkles size={16}/> Detect & classify angles</>}
          </button>
          {error && <div className="notice error" style={{ marginTop: 12 }}><AlertCircle size={15} style={{ verticalAlign: "-3px", marginRight: 5 }}/>{error}</div>}
        </div>
      </div>
    </div>

    {result && (
      <section style={{ marginTop: 22 }}>
        {/* Top Summary Metrics */}
        <div className="stats">
          <div className="stat">
            <div className="stat-label">Total Goats</div>
            <div className="stat-value">{result.goats_detected}</div>
            <div className="stat-note">Detected in image</div>
          </div>
          <div className="stat">
            <div className="stat-label">Valid Side Views</div>
            <div className="stat-value" style={{ color: "#00c853" }}>
              {result.valid_side_views ?? result.predictions.filter(p => p.is_valid_side).length}
            </div>
            <div className="stat-note">🟢 Ready for weight estimation</div>
          </div>
          <div className="stat">
            <div className="stat-label">Unsuitable Angles</div>
            <div className="stat-value" style={{ color: "#ef4444" }}>
              {result.invalid_views ?? result.predictions.filter(p => !p.is_valid_side).length}
            </div>
            <div className="stat-note">🔴 Frontal / rear / occluded</div>
          </div>
          <div className="stat">
            <div className="stat-label">Side Viability</div>
            <div className="stat-value">
              {result.goats_detected > 0
                ? `${Math.round(((result.valid_side_views ?? 0) / result.goats_detected) * 100)}%`
                : "0%"}
            </div>
            <div className="stat-note">{result.workflow || "Engine active"}</div>
          </div>
        </div>

        {/* Section Head with View Controls */}
        <div className="section-head" style={{ marginTop: 22 }}>
          <div>
            <h2>Side-Classification Visualization</h2>
            <p className="page-copy">
              Color-coded detection: 🟢 Green = Valid side view for weight | 🔴 Red = Unsuitable angle.
            </p>
          </div>
          {result.annotated_image && (
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className={`btn ${viewMode === "annotated" ? "btn-primary" : "btn-secondary"}`}
                style={{ padding: "6px 14px", fontSize: 12 }}
                onClick={() => setViewMode("annotated")}
              >
                Annotated Side View
              </button>
              <button
                type="button"
                className={`btn ${viewMode === "original" ? "btn-primary" : "btn-secondary"}`}
                style={{ padding: "6px 14px", fontSize: 12 }}
                onClick={() => setViewMode("original")}
              >
                Original Photo
              </button>
            </div>
          )}
        </div>

        {/* Full-width Output Image */}
        <div className="card" style={{ padding: 12, marginBottom: 22 }}>
          <div className="image-stage" style={{ width: "100%", minHeight: 380, display: "flex", justifyContent: "center", alignItems: "center", position: "relative", overflow: "hidden", borderRadius: 8, background: "#0b120e" }}>
            <img
              src={viewMode === "annotated" && result.annotated_image ? result.annotated_image : result.source_image}
              alt="Goat side classification output"
              style={{ maxWidth: "100%", maxHeight: 600, objectFit: "contain", borderRadius: 6, display: "block" }}
            />
            {/* If showing original without baked-in labels, overlay clean numeric chips */}
            {(viewMode === "original" || !result.annotated_image) && result.predictions.map((p, i) => {
              if (!p.box) return null;
              const cx = (p.box.x1 + p.box.x2) / 2;
              const top = Math.max(2, p.box.y1);
              return (
                <span
                  key={`${i}-${p.class}`}
                  className="id-chip"
                  style={{
                    position: "absolute",
                    left: `${(cx / result.image_width) * 100}%`,
                    top: `${(top / result.image_height) * 100}%`,
                    background: p.is_valid_side ? "#00c853" : "#ef4444",
                    color: "#fff",
                    fontWeight: 700,
                  }}
                >
                  #{idFor(i)}
                </span>
              );
            })}
          </div>
        </div>

        {/* Output Breakdown Located at the BOTTOM of the Image */}
        <div className="section-head">
          <div>
            <h2>Goat Side-View Breakdown ({result.goats_detected})</h2>
            <p className="page-copy">Detailed side quality percentage and weight feasibility for each detected goat.</p>
          </div>
        </div>

        {result.predictions.length === 0 ? (
          <div className="card empty" style={{ padding: 28, textAlign: "center", marginTop: 10 }}>
            No goats were detected in this image.
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14, marginTop: 12 }}>
            {result.predictions.map((p, i) => (
              <div
                className="result-card"
                key={i}
                style={{
                  margin: 0,
                  borderLeft: `4px solid ${p.is_valid_side ? "#00c853" : "#ef4444"}`,
                }}
              >
                <div className="result-top">
                  <h3>Goat #{idFor(i)}</h3>
                  <span
                    className="status"
                    style={{
                      background: p.is_valid_side ? "rgba(0, 200, 83, 0.15)" : "rgba(239, 68, 68, 0.15)",
                      color: p.is_valid_side ? "#00c853" : "#ef4444",
                      fontWeight: 700,
                      padding: "4px 8px",
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  >
                    {p.is_valid_side ? "VALID SIDE VIEW" : "INVALID ANGLE"}
                  </span>
                </div>

                <div
                  className="result-big"
                  style={{
                    color: p.is_valid_side ? "#00c853" : "#ef4444",
                    fontSize: 26,
                    margin: "8px 0 4px",
                  }}
                >
                  {p.side_view_score != null ? `${p.side_view_score}%` : "—"}
                  <span style={{ fontSize: 13, fontWeight: 500, color: "var(--fg-muted, #888)", marginLeft: 6 }}>
                    side quality
                  </span>
                </div>

                {/* Side Quality Progress Bar */}
                <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, marginBottom: 12, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, p.side_view_score || 0))}%`,
                      height: "100%",
                      background: p.is_valid_side ? "#00c853" : "#ef4444",
                      borderRadius: 3,
                    }}
                  />
                </div>

                <div className="result-grid">
                  <div className="kv">
                    <span>Weight Feasibility</span>
                    <b style={{ color: p.is_valid_side ? "#00c853" : "#ef4444" }}>
                      {p.is_valid_side ? "Ready for Weight" : "Not Usable"}
                    </b>
                  </div>
                  <div className="kv">
                    <span>Detection Conf</span>
                    <b>{p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : "—"}</b>
                  </div>
                  <div className="kv">
                    <span>Angle Category</span>
                    <b>{p.side_category || "Standard"}</b>
                  </div>
                  {p.box && (
                    <div className="kv" style={{ gridColumn: "1 / -1", fontSize: 11, color: "var(--fg-muted, #888)" }}>
                      <span>Coordinates</span>
                      <b>[{Math.round(p.box.x1)}, {Math.round(p.box.y1)}, {Math.round(p.box.x2)}, {Math.round(p.box.y2)}]</b>
                    </div>
                  )}
                </div>

                {/* Rejection / Angle Notes */}
                {p.rejection_reasons && p.rejection_reasons.length > 0 && (
                  <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.06)", fontSize: 11, color: "var(--fg-muted, #999)" }}>
                    <span style={{ fontWeight: 600, color: "#ef4444" }}>Notes: </span>
                    {p.rejection_reasons.join(", ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    )}
  </>;
}

