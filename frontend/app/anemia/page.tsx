"use client";

import { ChangeEvent, DragEvent, useEffect, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

type DetectorType = "roboflow" | "local" | "compare";

type ModelStatus = {
  available?: boolean;
  model?: string;
  task?: string;
  detectors?: {
    roboflow?: { available?: boolean; model_id?: string; provider?: string };
    local?: { available?: boolean; model_id?: string; provider?: string };
  };
  conjunctiva?: {
    configured?: boolean;
    roboflow_configured?: boolean;
    local_available?: boolean;
    model_id?: string;
  };
  famacha?: {
    configured?: boolean;
    model_id?: string | null;
    provider?: string;
  };
  message?: string;
};

type Prediction = {
  id?: number;
  class_id?: number | null;
  class_name?: string | null;
  confidence?: number | null;
  famacha_score?: number | null;
};

type ConjunctivaResult = {
  success?: boolean;
  detected?: boolean;
  model_id?: string;
  provider?: string;
  model_type?: string;
  prediction_count?: number;
  best_confidence?: number | null;
  latency_ms?: number | null;
  annotated_image?: string | null;
  error?: string | null;
  reason?: string | null;
};

type ComparisonEngineResult = {
  configured?: boolean;
  success?: boolean;
  detected?: boolean;
  model_id?: string;
  provider?: string;
  model_type?: string;
  prediction_count?: number;
  best_confidence?: number | null;
  latency_ms?: number | null;
  annotated_image?: string | null;
  error?: string | null;
  reason?: string | null;
};

type ComparisonData = {
  roboflow?: ComparisonEngineResult;
  local?: ComparisonEngineResult;
  agreement?: boolean;
};

type FamachaResult = {
  attempted?: boolean;
  success?: boolean;
  model_id?: string | null;
  provider?: string;
  score?: number | null;
  class_name?: string | null;
  confidence?: number | null;
  risk_level?: string | null;
  error?: string | null;
};

type AnemiaResult = {
  status?: string;
  detector_used?: string;
  conjunctiva?: ConjunctivaResult;
  famacha?: FamachaResult;
  comparison?: ComparisonData;
  model?: ModelStatus;
  predictions?: Prediction[];
  famacha_score?: number | null;
  risk_level?: string;
  annotated_image?: string;
  message?: string;
};

function scoreLabel(score: number | null | undefined) {
  if (!score) return "Unavailable";
  return `FAMACHA ${score}`;
}

function riskClass(risk: string | undefined) {
  if (!risk) return "neutral";
  if (risk.includes("Higher")) return "poor";
  if (risk.includes("Borderline")) return "acceptable";
  if (risk.includes("Lower")) return "good";
  return "neutral";
}

function getClinicalAdvice(score: number | null | undefined) {
  if (!score) return null;
  switch (score) {
    case 1:
      return {
        badge: "Optimal / No Anemia",
        desc: "Mucous membrane is healthy red. No deworming or treatment necessary.",
      };
    case 2:
      return {
        badge: "Acceptable / Normal",
        desc: "Red-pink membrane. Animal has good red blood cell levels; monitor routinely.",
      };
    case 3:
      return {
        badge: "Borderline / Warning",
        desc: "Pinkish membrane. Sub-optimal packed cell volume. Re-evaluate in 10-14 days.",
      };
    case 4:
      return {
        badge: "Dangerous Anemia",
        desc: "Pale pink to white membrane. High anemia risk; anthelmintic (dewormer) treatment strongly advised.",
      };
    case 5:
      return {
        badge: "Severe / Fatal Risk",
        desc: "Pure white / blanched membrane. Critical emergency; urgent veterinary intervention required.",
      };
    default:
      return null;
  }
}

export default function AnemiaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<AnemiaResult | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/anemia/status`)
      .then(async (r) => {
        if (!r.ok) throw new Error("Unable to read anemia model status.");
        return r.json();
      })
      .then(setModel)
      .catch(() =>
        setModel({
          available: false,
          message: "Backend anemia service is not reachable.",
        })
      );
  }, []);

  function chooseFile(next: File | null) {
    if (!next) return;

    const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
    if (!allowed.includes(next.type)) {
      setError("Please upload a JPG, PNG, BMP, or WEBP image.");
      return;
    }

    setFile(next);
    setResult(null);
    setError("");
    setPreview(URL.createObjectURL(next));
  }

  function onInput(e: ChangeEvent<HTMLInputElement>) {
    chooseFile(e.target.files?.[0] ?? null);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    chooseFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function analyze() {
    if (!file) {
      setError("Upload a goat eye image first.");
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);
      body.append("detector", "roboflow");

      const response = await fetch(`${API_BASE}/api/anemia/analyze`, {
        method: "POST",
        body,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || data?.message || "Anemia analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Anemia analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  const detected = result?.conjunctiva?.detected === true;
  const scored =
    detected &&
    result?.famacha?.success === true &&
    typeof result?.famacha?.score === "number";
  const score = scored ? (result?.famacha?.score ?? null) : null;
  const confidence =
    scored && typeof result?.famacha?.confidence === "number"
      ? result.famacha.confidence * 100
      : null;

  const clinical = getClinicalAdvice(score);

  return (
    <div>
      <div className="page-top">
        <div>
          <div className="eyebrow">LIVESTOCK ANEMIA / FAMACHA SCREENING</div>
          <h1>Goat FAMACHA Anemia Score</h1>
          <p className="page-copy">
            Upload a goat eye photo to determine the animal&apos;s FAMACHA anemia score
            (Score 1 to 5), anemia clinical risk, and eyelid conjunctiva detection.
          </p>
        </div>
      </div>

      {/* Top Status Cards */}
      <div className="stats">
        <div className="stat">
          <div className="stat-label">Conjunctiva Detection</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            AI Vision Model
          </div>
          <div className="stat-note">
            Instance Segmentation
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">FAMACHA Model</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            FAMACHA Classifier
          </div>
          <div className="stat-note">Clinical Scores 1–5</div>
        </div>

        <div className="stat">
          <div className="stat-label">Screening Status</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            {model?.available ? "Ready" : "Checking..."}
          </div>
          <div className="stat-note">Active Screening Pipeline</div>
        </div>

        <div className="stat">
          <div className="stat-label">Current FAMACHA Score</div>
          <div className="stat-value" style={{ fontSize: 18 }}>
            {score ? `Score ${score}` : "—"}
          </div>
          <div className="stat-note">
            {result?.risk_level || "Upload photo to score"}
          </div>
        </div>
      </div>

      {/* Upload and Preview Layout */}
      <div className="result-layout">
        <section className="card">
          <div className="section-head" style={{ marginTop: 0 }}>
            <div>
              <h2>1. Upload goat eye photo</h2>
              <p className="page-copy">
                Ensure the lower eyelid mucous membrane (conjunctiva) is clearly exposed and well-lit.
              </p>
            </div>
          </div>

          <div
            className={`dropzone ${dragging ? "is-dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              id="anemia-file"
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp"
              onChange={onInput}
              hidden
            />

            <label htmlFor="anemia-file" style={{ cursor: "pointer" }}>
              <div style={{ fontSize: 34 }}>◉</div>
              <div className="dropzone-title">Click to upload or drag and drop</div>
              <div className="dropzone-copy">JPG, JPEG, PNG, BMP or WEBP</div>
            </label>

            {file && (
              <div className="dropzone-file">
                <span>✓</span>
                <span>{file.name}</span>
              </div>
            )}
          </div>

          {error && (
            <div className="notice error" style={{ marginTop: 12 }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            style={{ width: "100%", marginTop: 14 }}
            onClick={analyze}
            disabled={!file || busy}
          >
            {busy ? "Analyzing FAMACHA..." : "Analyze FAMACHA Score"}
          </button>

          <div className="notice" style={{ marginTop: 12 }}>
            <b>Screening only:</b> AI FAMACHA scoring is a clinical screening aid for flock health, not a replacement for veterinary diagnosis.
          </div>
        </section>

        <section className="card">
          <div className="section-head" style={{ marginTop: 0 }}>
            <div>
              <h2>2. Uploaded image</h2>
              <p className="page-copy">Preview of the goat eye before screening.</p>
            </div>
          </div>

          <div className="preview" style={{ minHeight: 360 }}>
            {preview ? (
              <img src={preview} alt="Uploaded goat eye" />
            ) : (
              <div className="empty">
                <div style={{ fontSize: 32, marginBottom: 8 }}>◉</div>
                No goat eye image selected
              </div>
            )}
          </div>
        </section>
      </div>

      {/* FAMACHA RESULTS SECTION */}
      {result && (
        <section style={{ marginTop: 24 }}>
          <div>
            <div className="section-head">
              <div>
                <div className="eyebrow">SCREENING OUTCOME</div>
                <h2>FAMACHA Score Result</h2>
              </div>
              <span className={`status ${scored ? riskClass(result.risk_level) : detected ? "acceptable" : "poor"}`}>
                {scored ? result.risk_level : detected ? "Conjunctiva Detected" : "Conjunctiva Not Detected"}
              </span>
            </div>

            <div className="result-layout">
              {/* Left: Annotated Conjunctiva Image */}
              <div className="card">
                <div className="preview">
                  {result.annotated_image ? (
                    <img
                      src={result.annotated_image}
                      alt="Annotated goat eye conjunctiva"
                    />
                  ) : preview ? (
                    <img src={preview} alt="Goat eye" />
                  ) : (
                    <div className="empty">No image returned</div>
                  )}
                </div>
              </div>

              {/* Right: PRIMARY FAMACHA SCORE CARD */}
                <div className="card">
                  {scored ? (
                    <div>
                      <div className="result-card" style={{ borderColor: "var(--line)", boxShadow: "0 4px 14px rgba(0,0,0,0.04)" }}>
                        <div className="result-top">
                          <div>
                            <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--green)" }}>
                              DIAGNOSTIC SCORE
                            </span>
                            <h3 style={{ margin: "2px 0 0", fontSize: 18 }}>FAMACHA Score</h3>
                          </div>
                          <span className={`status ${riskClass(result.risk_level)}`} style={{ fontSize: 13, padding: "5px 10px" }}>
                            {scoreLabel(score)}
                          </span>
                        </div>

                        <div className="result-big" style={{ fontSize: 44, color: "var(--ink)", margin: "14px 0 4px", letterSpacing: "-0.03em" }}>
                          Score {score}
                        </div>

                        {clinical && (
                          <div style={{ marginBottom: 16, padding: "8px 12px", background: "var(--surface-soft)", borderRadius: 10, border: "1px solid var(--line)" }}>
                            <b style={{ color: "var(--ink)", fontSize: 13 }}>{clinical.badge}</b>
                            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{clinical.desc}</div>
                          </div>
                        )}

                        <div className="result-grid">
                          <div className="kv">
                            <span>Confidence</span>
                            <b>
                              {typeof confidence === "number"
                                ? `${confidence.toFixed(1)}%`
                                : "Unavailable"}
                            </b>
                          </div>

                          <div className="kv">
                            <span>Predicted Class</span>
                            <b>{result.famacha?.class_name || `FAMACHA_${score}`}</b>
                          </div>

                          <div className="kv">
                            <span>Anemia Risk</span>
                            <b style={{ color: riskClass(result.risk_level) === "poor" ? "var(--red)" : "inherit" }}>
                              {result.risk_level || "Unknown"}
                            </b>
                          </div>

                          <div className="kv">
                            <span>Conjunctiva Detector</span>
                            <b>Conjunctiva Segmentation Model</b>
                          </div>

                          <div className="kv">
                            <span>Detection Confidence</span>
                            <b>
                              {typeof result.conjunctiva?.best_confidence === "number"
                                ? `${(result.conjunctiva.best_confidence * 100).toFixed(1)}%`
                                : "—"}
                            </b>
                          </div>

                          <div className="kv">
                            <span>Scoring Model</span>
                            <b>FAMACHA Scoring Model (Scores 1–5)</b>
                          </div>
                        </div>
                      </div>

                      <div className="notice" style={{ marginTop: 12 }}>
                        {result.message || "FAMACHA score calculated successfully from detected conjunctiva."}
                      </div>
                    </div>
                  ) : (
                    /* Failure / Not Detected State */
                    <div>
                      <div className="result-card" style={{ borderColor: "#f3c2c2", background: "#fdf8f8" }}>
                        <div className="result-top">
                          <h3 style={{ margin: 0, color: "var(--red)" }}>FAMACHA Score</h3>
                          <span className="status poor">Not Generated</span>
                        </div>

                        <div className="result-big" style={{ color: "var(--muted)", fontSize: 32, margin: "14px 0 8px" }}>
                          No Score
                        </div>

                        <div style={{ fontSize: 13, color: "var(--ink)", lineHeight: 1.5, marginBottom: 12 }}>
                          {detected
                            ? "Conjunctiva was detected, but the FAMACHA scoring classifier could not evaluate the crop."
                            : "The goat's lower eyelid conjunctiva was not detected in this image. A valid conjunctiva region is required to assess mucous membrane color."}
                        </div>

                        <div className="result-grid">
                          <div className="kv">
                            <span>Conjunctiva Status</span>
                            <b>{detected ? "Detected" : "Not detected"}</b>
                          </div>
                          <div className="kv">
                            <span>Detector Engine</span>
                            <b>{result.detector_used || "Conjunctiva Segmentation Model"}</b>
                          </div>
                        </div>
                      </div>

                      <div className="notice error" style={{ marginTop: 12 }}>
                        <b>How to fix:</b> Upload a closer, well-lit photo where the animal&apos;s lower eyelid is gently pulled down to clearly reveal the mucous membrane.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
        </section>
      )}
    </div>
  );
}
