"use client";

import { ChangeEvent, DragEvent, useEffect, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

type ModelStatus = {
  available?: boolean;
  model?: string;
  path?: string;
  task?: string;
  classes?: Record<string, string>;
  message?: string;
};

type Prediction = {
  id?: number;
  class_id?: number | null;
  class_name?: string | null;
  confidence?: number | null;
  famacha_score?: number | null;
  polygon?: number[][];
};

type AnemiaResult = {
  status?: string;
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
        }),
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

  const prediction = result?.predictions?.[0];
  const score = result?.famacha_score ?? prediction?.famacha_score ?? null;
  const confidence = prediction?.confidence;

  return (
    <div>
      <div className="page-top">
        <div>
          <div className="eyebrow">ANEMIA / FAMACHA</div>
          <h1>Screen goat eye health.</h1>
          <p className="page-copy">
            Upload a clear lower-eyelid / conjunctiva image to screen the
            FAMACHA score. The AI result is a screening aid, not a veterinary
            diagnosis.
          </p>
        </div>
      </div>

      <div className="stats">
        <div className="stat">
          <div className="stat-label">Model</div>
          <div className="stat-value" style={{ fontSize: 20 }}>
            {model?.available ? "Ready" : "Offline"}
          </div>
          <div className="stat-note">{model?.model || "Checking backend..."}</div>
        </div>

        <div className="stat">
          <div className="stat-label">Task</div>
          <div className="stat-value" style={{ fontSize: 20 }}>
            FAMACHA
          </div>
          <div className="stat-note">
            {model?.task || "Conjunctiva screening"}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Scale</div>
          <div className="stat-value">1–5</div>
          <div className="stat-note">FAMACHA score</div>
        </div>

        <div className="stat">
          <div className="stat-label">Current result</div>
          <div className="stat-value" style={{ fontSize: 20 }}>
            {score ? `Score ${score}` : "—"}
          </div>
          <div className="stat-note">
            {result?.risk_level || "No analysis yet"}
          </div>
        </div>
      </div>

      <div className="result-layout">
        <section className="card">
          <div className="section-head" style={{ marginTop: 0 }}>
            <div>
              <h2>1. Upload goat eye photo</h2>
              <p className="page-copy">
                Use a close, well-lit image where the lower eyelid / conjunctiva
                is clearly visible.
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
              <div className="dropzone-title">
                Click to upload or drag and drop
              </div>
              <div className="dropzone-copy">
                JPG, JPEG, PNG, BMP or WEBP
              </div>
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
            {busy ? "Analyzing..." : "Analyze FAMACHA"}
          </button>

          <div className="notice" style={{ marginTop: 12 }}>
            <b>Screening only:</b> AI output should be reviewed by a qualified
            veterinarian before clinical decisions are made.
          </div>
        </section>

        <section className="card">
          <div className="section-head" style={{ marginTop: 0 }}>
            <div>
              <h2>2. Uploaded image</h2>
              <p className="page-copy">Preview before analysis.</p>
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

      {result && (
        <section style={{ marginTop: 22 }}>
          <div className="section-head">
            <div>
              <div className="eyebrow">ANALYSIS RESULT</div>
              <h2>FAMACHA screening result</h2>
            </div>
            <span className={`status ${riskClass(result.risk_level)}`}>
              {result.risk_level || "Result available"}
            </span>
          </div>

          <div className="result-layout">
            <div className="card">
              <div className="preview">
                {result.annotated_image ? (
                  <img
                    src={result.annotated_image}
                    alt="Annotated goat eye FAMACHA result"
                  />
                ) : preview ? (
                  <img src={preview} alt="Goat eye" />
                ) : (
                  <div className="empty">No result image returned.</div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="result-card">
                <div className="result-top">
                  <h3 style={{ margin: 0 }}>FAMACHA score</h3>
                  <span className={`status ${riskClass(result.risk_level)}`}>
                    {scoreLabel(score)}
                  </span>
                </div>

                <div className="result-big">
                  {score ? `Score ${score}` : "Unavailable"}
                </div>

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
                    <span>Class</span>
                    <b>{prediction?.class_name || "Unavailable"}</b>
                  </div>

                  <div className="kv">
                    <span>Risk</span>
                    <b>{result.risk_level || "Cannot determine"}</b>
                  </div>

                  <div className="kv">
                    <span>Model</span>
                    <b>{result.model?.model || "Unavailable"}</b>
                  </div>
                </div>
              </div>

              <div className="notice" style={{ marginTop: 12 }}>
                {result.message ||
                  "FAMACHA screening result returned by the GoatVision backend."}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
