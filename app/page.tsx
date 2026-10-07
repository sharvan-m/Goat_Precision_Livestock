import Link from "next/link";
import { ArrowRight, Camera, CircleCheck, Eye, IdCard, Scale, ShieldCheck, Video } from "lucide-react";
import { getJSON } from "@/lib/api";
import type { Health } from "@/lib/types";

async function safeHealth(): Promise<Health> {
  try {
    return await getJSON<Health>("/api/health");
  } catch {
    return { status: "offline", compute: "unknown", weight_model_loaded: false, weight_model_name: null, records: 0, bcs_available: false };
  }
}

export default async function Home() {
  const health = await safeHealth();

  return (
    <>
      <section className="hero">
        <div className="eyebrow">GOATVISION · SMART LIVESTOCK INTELLIGENCE</div>
        <h1>Know your goats<br />from the first frame.</h1>
        <p className="page-copy">
          A focused computer-vision workspace for goat detection, persistent IDs,
          suitable-view validation, morphometric features and weight estimation.
        </p>
        <div className="hero-actions">
          <Link className="btn btn-primary" href="/analyze"><Camera size={17}/> Analyze a goat <ArrowRight size={16}/></Link>
          <Link className="btn btn-secondary" href="/tracking"><Video size={17}/> Open live IDs</Link>
        </div>
      </section>

      <div className="stats">
        <div className="stat"><div className="stat-label">Saved records</div><div className="stat-value">{health.records}</div><div className="stat-note">Logged by the existing DataLogger</div></div>
        <div className="stat"><div className="stat-label">Compute</div><div className="stat-value">{health.compute === "cuda" ? "GPU" : "CPU"}</div><div className="stat-note">Current backend device</div></div>
        <div className="stat"><div className="stat-label">Weight model</div><div className="stat-value">{health.weight_model_loaded ? "Ready" : "Fallback"}</div><div className="stat-note">{health.weight_model_name || "No saved regressor"}</div></div>
        <div className="stat"><div className="stat-label">BCS model</div><div className="stat-value">Pending</div><div className="stat-note">Not connected in the current backend</div></div>
      </div>

      <div className="section-head"><div><h2>One workspace. Four decisions.</h2><p className="page-copy">The product experience follows the actual processing pipeline instead of turning every technical module into a separate screen.</p></div></div>
      <div className="grid-3">
        <div className="card feature-card"><div className="feature-icon"><Eye size={18}/></div><h3>Detect & validate</h3><p>Find goat candidates, inspect the view quality and make unsuitable frames visibly clear.</p></div>
        <div className="card feature-card"><div className="feature-icon"><IdCard size={18}/></div><h3>Track identity</h3><p>Keep a goat's ID consistent across video frames with the existing tracking workflows.</p></div>
        <div className="card feature-card"><div className="feature-icon"><Scale size={18}/></div><h3>Estimate weight</h3><p>Use extracted morphometric features and the configured regression model when the view is suitable.</p></div>
      </div>

      <div className="section-head"><div><h2>Designed for daily use</h2></div></div>
      <div className="grid-3">
        <Link href="/detect" className="card">
          <h3>🔎 Detect & count goats</h3>
          <p className="page-copy">Run your Roboflow segmentation workflow, count goat instances and inspect the returned annotation.</p>
        </Link>
        <Link href="/tracking" className="card">
          <h3>🆔 Live IDs</h3>
          <p className="page-copy">Track goats through video with stable identities and an operator-friendly tracking workflow.</p>
        </Link>
        <Link href="/records" className="card">
          <h3>📋 Herd records</h3>
          <p className="page-copy">Review stored predictions and export the records already produced by the system.</p>
        </Link>
      </div>

      <p className="footer-note">
        <ShieldCheck size={12} style={{verticalAlign:"-2px",marginRight:5}}/>
        Prediction outputs are presented as model outputs; field-validated accuracy requires calibrated imaging and independently measured ground truth.
      </p>
    </>
  );
}
