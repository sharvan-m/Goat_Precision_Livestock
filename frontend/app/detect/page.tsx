"use client";

import { useState, useCallback } from "react";
import { AlertCircle, CircleCheck, Loader2, ScanSearch, Sparkles, RotateCcw, Sliders, Check } from "lucide-react";
import { postForm } from "@/lib/api";
import { UploadDropzone } from "@/components/UploadDropzone";
import { useAnalysis, DetectResponse } from "@/lib/AnalysisContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function GoatDetectionPage() {
  const { detect, setDetect, resetDetect } = useAnalysis();
  const { file, preview, result, busy, error, viewMode } = detect;
  const [precisionMode, setPrecisionMode] = useState(false);
  const [recalibratingId, setRecalibratingId] = useState<number | null>(null);
  const [customHeights, setCustomHeights] = useState<Record<number, string>>({});

  const setFile = useCallback(
    (f: File | null) => {
      if (preview) {
        try {
          URL.revokeObjectURL(preview);
        } catch {}
      }
      setDetect({
        file: f,
        preview: f ? URL.createObjectURL(f) : "",
        result: null,
        error: "",
      });
    },
    [preview, setDetect]
  );

  async function scan() {
    if (!file) return;
    setDetect({ busy: true, error: "", result: null });
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("detector", "roboflow");
      const res = await postForm<DetectResponse>("/api/detect/goats", form);
      setDetect({ result: res, busy: false });
    } catch (e) {
      setDetect({ error: e instanceof Error ? e.message : "Goat detection failed.", busy: false });
    }
  }

  async function handleRecalibrate(goatId: number, cohort: string, customHeight?: number) {
    if (!result) return;
    const predIndex = result.predictions.findIndex((p) => (p.id ?? 0) === goatId);
    if (predIndex === -1) return;
    const pred = result.predictions[predIndex];
    if (!pred.measurements) return;

    setRecalibratingId(goatId);
    try {
      const res = await fetch(`${API}/api/detect/recalibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goat_id: goatId,
          cohort: cohort,
          custom_height_cm: customHeight,
          measurements: pred.measurements,
        }),
      });
      if (!res.ok) throw new Error("Recalibration failed");
      const data = await res.json();

      const updated = [...result.predictions];
      updated[predIndex] = {
        ...pred,
        weight_kg: data.weight_kg,
        weight_margin_kg: data.weight_margin_kg,
        bcs_score: data.bcs_score,
        bcs_label: data.bcs_label,
        cohort: data.cohort,
        withers_height_cm: data.withers_height_cm,
        measurements: data.measurements,
      };

      setDetect({
        result: {
          ...result,
          predictions: updated,
        },
      });
    } catch (err) {
      console.error("Recalibrate error:", err);
    } finally {
      setRecalibratingId(null);
    }
  }

  async function handleBatchCohort(cohort: string) {
    if (!result) return;
    const valids = result.predictions.filter((p) => p.is_valid_side && p.measurements && p.id);
    for (const g of valids) {
      await handleRecalibrate(g.id!, cohort);
    }
  }

  return <>
    <div className="page-top">
      <div>
        <div className="eyebrow">ANALYZE · VISION BIOMETRICS</div>
        <h1>Analyze: Detection, Weight & BCS</h1>
        <p className="page-copy">
          Detects every goat in the frame, validates side-profile angle feasibility, and calculates live weight (kg) and Body Condition Score (BCS).
        </p>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {(file || result) && (
          <button
            className="btn btn-ghost"
            style={{ padding: "7px 14px", fontSize: 12 }}
            onClick={resetDetect}
            title="Clear and start fresh"
          >
            <RotateCcw size={13} /> Clear
          </button>
        )}
        <div className="status good">
          <CircleCheck size={14}/> Active
        </div>
      </div>
    </div>

    <div className="result-layout">
      <section className="card">
        <div className="section-head" style={{ marginTop: 0 }}>
          <div>
            <h2>1. Upload goat image</h2>
            <p className="page-copy">
              Upload a clear side-view photo of goats for weight and body condition assessment.
            </p>
          </div>
        </div>

        <UploadDropzone file={file} onFile={setFile}/>

        <button
          className="btn btn-primary"
          disabled={!file || busy}
          style={{ width: "100%", marginTop: 14, opacity: (!file || busy) ? 0.55 : 1 }}
          onClick={scan}
        >
          {busy ? <><Loader2 size={16} className="spin"/> Analyzing image…</> : <><Sparkles size={16}/> Analyze Photo</>}
        </button>

        {error && (
          <div className="notice error" style={{ marginTop: 12 }}>
            <AlertCircle size={15} style={{ verticalAlign: "-3px", marginRight: 5 }}/>
            {error}
          </div>
        )}

        <div className="feature-card" style={{ marginTop: 14, minHeight: "auto", padding: "12px 14px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div className="feature-icon" style={{ width: 26, height: 26, borderRadius: 8, margin: 0 }}><ScanSearch size={14}/></div>
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 700 }}>Side-View Quality & Classification</h3>
          </div>
          <p className="page-copy" style={{ fontSize: 11, lineHeight: 1.45 }}>
            🟢 <b>LIGHT GREEN</b>: Flank & torso visible at correct side angle (Weight & BCS calculated).
            <br/>
            🔴 <b>RED</b>: Front/rear-facing, grazing, or distant profile (Weight gated).
          </p>
        </div>
      </section>

      <section className="card">
        <div className="section-head" style={{ marginTop: 0 }}>
          <div>
            <h2>2. Uploaded image</h2>
            <p className="page-copy">Preview of the goat photo before analysis.</p>
          </div>
        </div>

        <div className="preview" style={{ minHeight: 360 }}>
          {preview ? (
            <img src={preview} alt="Uploaded goat" />
          ) : (
            <div className="empty">
              <div style={{ fontSize: 32, marginBottom: 8 }}>◉</div>
              No goat image selected
            </div>
          )}
        </div>
      </section>
    </div>

    {/* Limitations & Requirements of Weight Estimation Box */}
    <div className="card" style={{ marginTop: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <div style={{ width: 30, height: 30, borderRadius: 9, background: "#fef3c7", display: "grid", placeItems: "center", flexShrink: 0 }}>
          <AlertCircle size={17} color="#b45309" />
        </div>
        <div>
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 800, color: "var(--ink)", letterSpacing: "-0.01em" }}>
            Physical Requirements & Limitations for Weight Estimation
          </h3>
          <p style={{ margin: 0, fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
            2D computer vision biometrics require specific posture and visibility to accurately calculate live weight (±2.5 kg).
          </p>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginTop: 14 }}>
        <div style={{ padding: "12px 14px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 12 }}>
          <b style={{ color: "var(--green-dark, #1f5834)", fontSize: 12, display: "block", marginBottom: 4, fontWeight: 750 }}>
            1. 90° Full Side Profile
          </b>
          <p style={{ margin: 0, fontSize: 11, color: "var(--muted, #718078)", lineHeight: 1.45 }}>
            Entire flank and barrel must face the camera. Frontal, rear, or angled perspectives cannot measure barrel depth.
          </p>
        </div>

        <div style={{ padding: "12px 14px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 12 }}>
          <b style={{ color: "var(--green-dark, #1f5834)", fontSize: 12, display: "block", marginBottom: 4, fontWeight: 750 }}>
            2. Standing Upright (Not Grazing)
          </b>
          <p style={{ margin: 0, fontSize: 11, color: "var(--muted, #718078)", lineHeight: 1.45 }}>
            Head must be raised or neutral. Grazing (head to ground) stretches the neck and falsely distorts body length.
          </p>
        </div>

        <div style={{ padding: "12px 14px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 12 }}>
          <b style={{ color: "var(--green-dark, #1f5834)", fontSize: 12, display: "block", marginBottom: 4, fontWeight: 750 }}>
            3. Full Hoof & Leg Visibility
          </b>
          <p style={{ margin: 0, fontSize: 11, color: "var(--muted, #718078)", lineHeight: 1.45 }}>
            Legs must not be buried in tall pasture grass or mud, which corrupts the stature scale calibration.
          </p>
        </div>

        <div style={{ padding: "12px 14px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 12 }}>
          <b style={{ color: "var(--green-dark, #1f5834)", fontSize: 12, display: "block", marginBottom: 4, fontWeight: 750 }}>
            4. Foreground Proximity (&gt;72px)
          </b>
          <p style={{ margin: 0, fontSize: 11, color: "var(--muted, #718078)", lineHeight: 1.45 }}>
            Animal must be in the foreground. Distant background goats are gated out to prevent pixel scaling errors.
          </p>
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
                onClick={() => setDetect({ viewMode: "annotated" })}
              >
                Annotated Side View
              </button>
              <button
                type="button"
                className={`btn ${viewMode === "original" ? "btn-primary" : "btn-secondary"}`}
                style={{ padding: "6px 14px", fontSize: 12 }}
                onClick={() => setDetect({ viewMode: "original" })}
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
        <div className="section-head" style={{ alignItems: "center" }}>
          <div>
            <h2>Goat Side-View Breakdown ({result.goats_detected})</h2>
            <p className="page-copy">Detailed side quality percentage and weight feasibility for each detected goat.</p>
          </div>

          {/* Precision Calibration Toggle */}
          <button
            type="button"
            className={`btn ${precisionMode ? "btn-primary" : "btn-secondary"}`}
            style={{ fontSize: 12, padding: "7px 14px", display: "inline-flex", alignItems: "center", gap: 6 }}
            onClick={() => setPrecisionMode(!precisionMode)}
          >
            <Sliders size={14} />
            <span>{precisionMode ? "🔬 Precision Calibrator Active" : "🔬 Enable Precision Calibration"}</span>
          </button>
        </div>

        {/* Precision Calibration Explanatory Banner & Batch Bar */}
        {precisionMode && (
          <div className="card" style={{ padding: 14, marginBottom: 14, background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 14 }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div>
                <b style={{ color: "var(--green-dark, #1f5834)", fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
                  <Sliders size={14} /> Precision Calibration: Per-Goat Life-Stage Stature Tuning
                </b>
                <p style={{ margin: "4px 0 0", fontSize: 11, color: "var(--muted)", maxWidth: 680, lineHeight: 1.45 }}>
                  Because 2D photos lack laser depth, a 15 kg young goat up close has the same pixel height as a 40 kg adult further away.
                  Select the life-stage cohort or enter measured withers height (cm) for individual goats below to achieve laboratory accuracy (±1.2 kg).
                </p>
              </div>

              {/* Batch Life-Stage Shortcut Bar */}
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", marginRight: 2 }}>Batch Apply:</span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: 10, padding: "4px 10px", borderRadius: 8 }}
                  onClick={() => handleBatchCohort("kid")}
                  title="Under 6 months (~38 cm height, 8-15 kg)"
                >
                  All Kids (&lt;6m)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: 10, padding: "4px 10px", borderRadius: 8 }}
                  onClick={() => handleBatchCohort("yearling")}
                  title="6 to 14 months (~46 cm height, 15-25 kg)"
                >
                  All Yearlings (6-14m)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: 10, padding: "4px 10px", borderRadius: 8 }}
                  onClick={() => handleBatchCohort("adult")}
                  title="Mature adult domestic (~62 cm height, 30-45 kg)"
                >
                  All Adults (14m+)
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: 10, padding: "4px 10px", borderRadius: 8 }}
                  onClick={() => handleBatchCohort("heavy")}
                  title="Boer / meat buck (~74 cm height, 50-80 kg)"
                >
                  All Heavy Boer
                </button>
              </div>
            </div>
          </div>
        )}

        {result.predictions.length === 0 ? (
          <div className="card empty" style={{ padding: 28, textAlign: "center", marginTop: 10 }}>
            No goats were detected in this image.
          </div>
        ) : (
          <div className="goat-result-grid" style={{ marginTop: 12 }}>
            {result.predictions.map((p, i) => {
              /* Compute an overall quality score 0-100 */
              const sideScore = p.side_view_score ?? 0;
              const confScore = p.confidence != null ? p.confidence * 100 : 0;
              const visScore = p.visibility_pct ?? 100;
              const qualityScore = Math.round(sideScore * 0.45 + confScore * 0.30 + visScore * 0.25);
              const qualityLabel = qualityScore >= 70 ? "ACCEPTABLE" : qualityScore >= 45 ? "ACCEPTABLE" : "POOR";
              const qualityColor = qualityScore >= 70 ? "#27683d" : qualityScore >= 45 ? "#8c611b" : "#a33a3a";
              const qualityBg = qualityScore >= 70 ? "#e7f5ea" : qualityScore >= 45 ? "#fff0d2" : "#ffe1e1";
              const confLabel = confScore >= 85 ? "HIGH" : confScore >= 60 ? "MEDIUM" : "LOW";
              const isRecalibrating = recalibratingId === p.id;
              const activeCohort = p.cohort || "adult";

              return (
                <div
                  className="result-card"
                  key={i}
                  style={{ margin: 0, display: "flex", flexDirection: "column" }}
                >
                  {/* ── Header: Number + Name + Quality Badge ── */}
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 10 }}>
                    <div className="goat-name">
                      <span className="goat-number">{String(i + 1).padStart(2, "0")}</span>
                      <h3>Goat {String(i + 1).padStart(2, "0")}</h3>
                    </div>
                    <span
                      className="status"
                      style={{
                        background: qualityBg,
                        color: qualityColor,
                        fontWeight: 800,
                        padding: "4px 10px",
                        borderRadius: 999,
                        fontSize: 10,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {qualityLabel} · {qualityScore}/100
                    </span>
                  </div>

                  {/* ── Weight or Unavailable ── */}
                  {p.weight_kg != null ? (
                    <div style={{ marginBottom: 6 }}>
                      <div style={{ fontSize: 28, fontWeight: 850, color: "var(--ink)", letterSpacing: "-0.03em", lineHeight: 1.1, display: "flex", alignItems: "baseline", gap: 6 }}>
                        {isRecalibrating ? (
                          <span style={{ fontSize: 20, color: "var(--muted)" }}><Loader2 size={18} className="spin" style={{ display: "inline-block", verticalAlign: "-2px" }} /> Updating…</span>
                        ) : (
                          <>
                            {p.weight_kg.toFixed(1)} <span style={{ fontSize: 16, fontWeight: 700 }}>kg</span>
                          </>
                        )}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 3 }}>
                        Confidence: {confLabel}
                        {p.withers_height_cm && (
                          <span style={{ marginLeft: 6, fontWeight: 600, color: "var(--green-dark)" }}>
                            · Prior: {Math.round(p.withers_height_cm)} cm
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div style={{ marginBottom: 6 }}>
                      <div style={{ fontSize: 22, fontWeight: 850, color: "var(--ink)", letterSpacing: "-0.02em", lineHeight: 1.15 }}>
                        Weight unavailable
                      </div>
                      <div className="result-subline" style={{ color: "var(--muted)", marginTop: 4 }}>
                        {p.rejection_reasons && p.rejection_reasons.length > 0 ? p.rejection_reasons.join(". ") : "Angle or distance unsuitable for biometric weighing."}
                      </div>
                    </div>
                  )}

                  {/* ── BCS Badge (if available) ── */}
                  {p.bcs_score != null && (
                    <div className="bcs-row" style={{ marginBottom: 10 }}>
                      <span className="bcs-badge">BCS {p.bcs_score.toFixed(1)}</span>
                      <span className="bcs-label" style={{ color: "var(--muted)" }}>{p.bcs_label}</span>
                    </div>
                  )}

                  {/* ── Precision Calibrator Selector (Per Goat) ── */}
                  {precisionMode && p.is_valid_side && p.measurements && (
                    <div style={{ margin: "6px 0 10px", padding: "8px 9px", background: "var(--surface-soft, #f8faf7)", border: "1px solid var(--line, #dfe7e0)", borderRadius: 10 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 10, fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--muted)", marginBottom: 6 }}>
                        <span>Cohort Calibration</span>
                        <span style={{ color: "var(--green-dark)" }}>
                          {activeCohort === "kid" ? "Kid (<6m)" : activeCohort === "yearling" || activeCohort === "young" ? "Yearling (6-14m)" : activeCohort === "heavy" ? "Boer / Heavy" : activeCohort === "custom" ? "Custom Height" : "Adult (14m+)"}
                        </span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                        <button
                          type="button"
                          className={`btn ${activeCohort === "kid" ? "btn-primary" : "btn-secondary"}`}
                          style={{ fontSize: 9, padding: "5px 4px", justifyContent: "center" }}
                          onClick={() => handleRecalibrate(p.id!, "kid")}
                          disabled={isRecalibrating}
                        >
                          Kid (&lt;6m · 38cm)
                        </button>
                        <button
                          type="button"
                          className={`btn ${activeCohort === "yearling" || activeCohort === "young" ? "btn-primary" : "btn-secondary"}`}
                          style={{ fontSize: 9, padding: "5px 4px", justifyContent: "center" }}
                          onClick={() => handleRecalibrate(p.id!, "yearling")}
                          disabled={isRecalibrating}
                        >
                          Yearling (6-14m · 46cm)
                        </button>
                        <button
                          type="button"
                          className={`btn ${activeCohort === "adult" && !customHeights[p.id!] ? "btn-primary" : "btn-secondary"}`}
                          style={{ fontSize: 9, padding: "5px 4px", justifyContent: "center" }}
                          onClick={() => handleRecalibrate(p.id!, "adult")}
                          disabled={isRecalibrating}
                        >
                          Adult (14m+ · 62cm)
                        </button>
                        <button
                          type="button"
                          className={`btn ${activeCohort === "heavy" ? "btn-primary" : "btn-secondary"}`}
                          style={{ fontSize: 9, padding: "5px 4px", justifyContent: "center" }}
                          onClick={() => handleRecalibrate(p.id!, "heavy")}
                          disabled={isRecalibrating}
                        >
                          Boer Buck (74cm)
                        </button>
                      </div>

                      {/* Manual withers height input */}
                      <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 6, paddingTop: 6, borderTop: "1px dashed var(--line)" }}>
                        <span style={{ fontSize: 10, color: "var(--muted)" }}>Custom height:</span>
                        <input
                          type="number"
                          min={25}
                          max={95}
                          placeholder={String(p.withers_height_cm ? Math.round(p.withers_height_cm) : 45)}
                          value={customHeights[p.id!] || ""}
                          onChange={(e) => setCustomHeights({ ...customHeights, [p.id!]: e.target.value })}
                          style={{ width: 50, padding: "2px 5px", fontSize: 11, borderRadius: 6, border: "1px solid var(--line)" }}
                        />
                        <span style={{ fontSize: 10, color: "var(--muted)" }}>cm</span>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          style={{ fontSize: 10, padding: "2px 7px", marginLeft: "auto" }}
                          disabled={!customHeights[p.id!] || isRecalibrating}
                          onClick={() => {
                            const val = parseFloat(customHeights[p.id!]);
                            if (!isNaN(val) && val > 20) {
                              handleRecalibrate(p.id!, "custom", val);
                            }
                          }}
                        >
                          Apply
                        </button>
                      </div>
                    </div>
                  )}

                  {/* ── 2×2 Stats Grid ── */}
                  <div className="result-grid" style={{ flex: 1 }}>
                    <div className="kv">
                      <span>Detection</span>
                      <b>{p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : "—"}</b>
                    </div>
                    <div className="kv">
                      <span>Side View</span>
                      <b style={{ color: p.is_valid_side ? "#27683d" : "#a33a3a" }}>
                        {p.is_valid_side ? "SIDE" : "NOT SIDE"}
                      </b>
                    </div>
                    <div className="kv">
                      <span>Side Confidence</span>
                      <b>{sideScore > 0 ? `${sideScore}%` : "—"}</b>
                    </div>
                    <div className="kv">
                      <span>Estimation</span>
                      <b style={{ color: p.is_valid_side ? "#27683d" : "#a33a3a" }}>
                        {p.is_valid_side ? "Allowed" : "Blocked"}
                      </b>
                    </div>
                  </div>

                  {/* ── Morphometric Features (collapsible) ── */}
                  <details className="measure-details" style={{ marginTop: 8 }}>
                    <summary>▸ Morphometric Features</summary>
                    <div className="code-block" style={{ marginTop: 8, fontSize: 10, lineHeight: 1.65 }}>
                      <div>Angle Category: {p.side_category || "Standard"}</div>
                      <div>Weight Feasibility: {p.is_valid_side ? "Ready for Weight" : "Not Usable"}</div>
                      <div>Body Visibility: {p.visibility_pct != null ? `${p.visibility_pct}%` : "100%"}</div>
                      {p.cohort && <div>Calibrated Cohort: {p.cohort}</div>}
                      {p.withers_height_cm && <div>Withers Stature: {Math.round(p.withers_height_cm)} cm</div>}
                      {p.measurements?.body_length && <div>Body Length: {p.measurements.body_length} cm</div>}
                      {p.measurements?.girth && <div>Heart Girth: {p.measurements.girth} cm</div>}
                      {p.weight_kg != null && <div>Live Weight: {p.weight_kg.toFixed(1)} kg</div>}
                      {p.bcs_score != null && <div>BCS: {p.bcs_score.toFixed(1)} ({p.bcs_label})</div>}
                      {p.weight_margin_kg != null && <div>Weight Margin: ±{p.weight_margin_kg} kg (95% CI)</div>}
                      {p.box && <div>Coordinates: [{Math.round(p.box.x1)}, {Math.round(p.box.y1)}, {Math.round(p.box.x2)}, {Math.round(p.box.y2)}]</div>}
                      {p.rejection_reasons && p.rejection_reasons.length > 0 && <div style={{ color: "#ef4444" }}>Notes: {p.rejection_reasons.join(", ")}</div>}
                    </div>
                  </details>
                </div>
              );
            })}
          </div>
        )}
      </section>
    )}
  </>;
}
