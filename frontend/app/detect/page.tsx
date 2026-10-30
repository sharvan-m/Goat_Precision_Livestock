"use client";

import { useState } from "react";
import { AlertCircle, CircleCheck, Loader2, ScanSearch, Sparkles } from "lucide-react";
import { postForm } from "@/lib/api";
import { UploadDropzone } from "@/components/UploadDropzone";

type Prediction = {
  id?: number;
  class: string;
  confidence: number | null;
  box: { x1: number; y1: number; x2: number; y2: number } | null;
  center?: { x: number; y: number } | null;
  side_view_score?: number;
  visibility_pct?: number;
  is_valid_side?: boolean;
  side_category?: string;
  rejection_reasons?: string[];
  weight_kg?: number | null;
  weight_margin_kg?: number | null;
  bcs_score?: number | null;
  bcs_label?: string | null;
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

export default function GoatDetectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Response | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState<"annotated" | "original">("annotated");

  async function scan() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("detector", "roboflow");
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
          Valid side angles are annotated in <b>LIGHT GREEN</b>; invalid/other angles in <b>RED</b>.
        </p>
      </div>
      <div className="status good">
        <CircleCheck size={14}/> Active
      </div>
    </div>

    <div className="card">
      <div className="grid-2">
        <div>
          <UploadDropzone file={file} onFile={setFile}/>
        </div>
        <div>
          <div className="feature-card" style={{ marginBottom: 14 }}>
            <div className="feature-icon"><ScanSearch size={17}/></div>
            <h3>Side-View Quality Classification</h3>
            <p className="page-copy">
              Color-codes bounding masks with centered ID numbers (1, 2, 3...):
              <br/>
              🟢 <b>LIGHT GREEN</b>: Flank & torso visible at the correct side angle (Weight estimation eligible).
              <br/>
              🔴 <b>RED</b>: Front-facing, rear-facing, or non-side profile.
            </p>
          </div>
          <button
            className="btn btn-primary"
            disabled={!file || busy}
            style={{ width: "100%", opacity: (!file || busy) ? 0.55 : 1 }}
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
            <div className="stat-value" style={{ color: "#10b981" }}>
              {result.valid_side_views ?? result.predictions.filter(p => p.is_valid_side).length}
            </div>
            <div className="stat-note">🟢 Ready for weight estimation</div>
          </div>
          <div className="stat">
            <div className="stat-label">Unsuitable Angles</div>
            <div className="stat-value" style={{ color: "#ef4444" }}>
              {result.invalid_views ?? result.predictions.filter(p => !p.is_valid_side).length}
            </div>
            <div className="stat-note">🔴 Frontal / rear / non-side</div>
          </div>
          <div className="stat">
            <div className="stat-label">Side Viability</div>
            <div className="stat-value">
              {result.goats_detected > 0
                ? `${Math.round(((result.valid_side_views ?? 0) / result.goats_detected) * 100)}%`
                : "0%"}
            </div>
            <div className="stat-note">GoatVision AI Vision Pipeline</div>
          </div>
        </div>

        {/* Section Head with View Controls */}
        <div className="section-head" style={{ marginTop: 22 }}>
          <div>
            <h2>Side-Classification Visualization</h2>
            <p className="page-copy">
              Color-coded detection: 🟢 Light Green = Valid side view for weight | 🔴 Red = Invalid angle.
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
            {/* If showing original without baked-in labels, overlay clean numeric chips at center of body */}
            {(viewMode === "original" || !result.annotated_image) && result.predictions.map((p, i) => {
              if (!p.box) return null;
              const cx = p.center?.x ?? (p.box.x1 + p.box.x2) / 2;
              const cy = p.center?.y ?? (p.box.y1 + p.box.y2) / 2;
              return (
                <span
                  key={`${i}-${p.class}`}
                  className="id-chip"
                  style={{
                    position: "absolute",
                    left: `${(cx / result.image_width) * 100}%`,
                    top: `${(cy / result.image_height) * 100}%`,
                    transform: "translate(-50%, -50%)",
                    background: "rgba(20, 20, 20, 0.9)",
                    border: `2px solid ${p.is_valid_side ? "#10b981" : "#ef4444"}`,
                    color: "#fff",
                    fontWeight: 800,
                    width: 26,
                    height: 26,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    boxShadow: "0 2px 6px rgba(0,0,0,0.5)",
                  }}
                >
                  {i + 1}
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
                  borderLeft: `4px solid ${p.is_valid_side ? "#10b981" : "#ef4444"}`,
                }}
              >
                <div className="result-top">
                  <h3>Goat {i + 1}</h3>
                  <span
                    className="status"
                    style={{
                      background: p.is_valid_side ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                      color: p.is_valid_side ? "#10b981" : "#ef4444",
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
                    color: p.is_valid_side ? "#10b981" : "#ef4444",
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
                <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, marginBottom: 10, overflow: "hidden" }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, p.side_view_score || 0))}%`,
                      height: "100%",
                      background: p.is_valid_side ? "#10b981" : "#ef4444",
                      borderRadius: 3,
                    }}
                  />
                </div>

                {/* Live Weight & BCS Display */}
                {p.weight_kg != null ? (
                  <div style={{ margin: "10px 0 12px", padding: "10px 12px", background: "rgba(16, 185, 129, 0.08)", border: "1px solid rgba(16, 185, 129, 0.25)", borderRadius: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                      <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-muted, #888)", fontWeight: 700 }}>Estimated Live Weight</span>
                      {p.weight_margin_kg != null && <span style={{ fontSize: 11, color: "var(--fg-muted, #888)" }}>± {p.weight_margin_kg} kg (95% CI)</span>}
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 850, color: "#10b981", margin: "2px 0 6px" }}>
                      {p.weight_kg.toFixed(1)} <span style={{ fontSize: 15, fontWeight: 600 }}>kg</span>
                    </div>
                    {p.bcs_score != null && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, paddingTop: 6, borderTop: "1px solid rgba(16, 185, 129, 0.15)" }}>
                        <span style={{ fontSize: 11, fontWeight: 800, padding: "2px 8px", borderRadius: 999, background: "#10b981", color: "#062817" }}>
                          BCS {p.bcs_score.toFixed(1)}
                        </span>
                        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--fg, #e2e8f0)" }}>
                          {p.bcs_label}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ margin: "10px 0 12px", padding: "8px 10px", background: "rgba(239, 68, 68, 0.06)", border: "1px solid rgba(239, 68, 68, 0.15)", borderRadius: 8, fontSize: 12, color: "var(--fg-muted, #888)" }}>
                    <span style={{ fontWeight: 600, color: "#ef4444" }}>Weight gated: </span>
                    Side profile not optimal for exact biometric scale
                  </div>
                )}

                <div className="result-grid">
                  <div className="kv">
                    <span>Live Weight</span>
                    <b style={{ color: p.weight_kg != null ? "#10b981" : "#ef4444" }}>
                      {p.weight_kg != null ? `${p.weight_kg.toFixed(1)} kg` : "Angle Blocked"}
                    </b>
                  </div>
                  <div className="kv">
                    <span>Body Condition</span>
                    <b style={{ color: p.bcs_score != null ? "#10b981" : "inherit" }}>
                      {p.bcs_score != null ? `BCS ${p.bcs_score.toFixed(1)}` : "—"}
                    </b>
                  </div>
                  <div className="kv">
                    <span>Weight Feasibility</span>
                    <b style={{ color: p.is_valid_side ? "#10b981" : "#ef4444" }}>
                      {p.is_valid_side ? "Ready for Weight" : "Not Usable"}
                    </b>
                  </div>
                  <div className="kv">
                    <span>Body Visibility</span>
                    <b style={{ color: (p.visibility_pct ?? 100) >= 70 ? "#10b981" : "#ef4444" }}>
                      {p.visibility_pct != null ? `${p.visibility_pct}%` : "100%"}
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

