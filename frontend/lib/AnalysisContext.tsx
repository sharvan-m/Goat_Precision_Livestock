"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";

/* ────────────────────────────────────────────────────────
   Shared types for Detect (Analyze) page
   ──────────────────────────────────────────────────────── */

export type DetectPrediction = {
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

export type DetectResponse = {
  source_image: string;
  annotated_image: string | null;
  goats_detected: number;
  valid_side_views?: number;
  invalid_views?: number;
  predictions: DetectPrediction[];
  workspace: string;
  workflow: string;
  image_width: number;
  image_height: number;
  annotation_source?: string;
};

/* ────────────────────────────────────────────────────────
   Shared types for Anemia page
   ──────────────────────────────────────────────────────── */

export type AnemiaModelStatus = {
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

export type AnemiaConjunctivaResult = {
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

export type AnemiaComparisonEngineResult = {
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

export type AnemiaFamachaResult = {
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

export type AnemiaPrediction = {
  id?: number;
  class_id?: number | null;
  class_name?: string | null;
  confidence?: number | null;
  famacha_score?: number | null;
};

export type AnemiaResult = {
  status?: string;
  detector_used?: string;
  conjunctiva?: AnemiaConjunctivaResult;
  famacha?: AnemiaFamachaResult;
  comparison?: {
    roboflow?: AnemiaComparisonEngineResult;
    local?: AnemiaComparisonEngineResult;
    agreement?: boolean;
  };
  model?: AnemiaModelStatus;
  predictions?: AnemiaPrediction[];
  famacha_score?: number | null;
  risk_level?: string;
  annotated_image?: string;
  message?: string;
};

/* ────────────────────────────────────────────────────────
   Context shape
   ──────────────────────────────────────────────────────── */

type DetectState = {
  file: File | null;
  result: DetectResponse | null;
  busy: boolean;
  error: string;
  viewMode: "annotated" | "original";
};

type AnemiaState = {
  file: File | null;
  preview: string;
  result: AnemiaResult | null;
  model: AnemiaModelStatus | null;
  busy: boolean;
  error: string;
};

type AnalysisContextType = {
  detect: DetectState;
  setDetect: (updater: Partial<DetectState> | ((prev: DetectState) => DetectState)) => void;
  resetDetect: () => void;

  anemia: AnemiaState;
  setAnemia: (updater: Partial<AnemiaState> | ((prev: AnemiaState) => AnemiaState)) => void;
  resetAnemia: () => void;
};

const defaultDetect: DetectState = {
  file: null,
  result: null,
  busy: false,
  error: "",
  viewMode: "annotated",
};

const defaultAnemia: AnemiaState = {
  file: null,
  preview: "",
  result: null,
  model: null,
  busy: false,
  error: "",
};

const AnalysisContext = createContext<AnalysisContextType | null>(null);

/* ────────────────────────────────────────────────────────
   Provider — wraps layout so state survives navigation
   ──────────────────────────────────────────────────────── */

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [detect, setDetectRaw] = useState<DetectState>(defaultDetect);
  const [anemia, setAnemiaRaw] = useState<AnemiaState>(defaultAnemia);

  const setDetect = useCallback(
    (updater: Partial<DetectState> | ((prev: DetectState) => DetectState)) => {
      setDetectRaw((prev) =>
        typeof updater === "function" ? updater(prev) : { ...prev, ...updater }
      );
    },
    []
  );

  const setAnemia = useCallback(
    (updater: Partial<AnemiaState> | ((prev: AnemiaState) => AnemiaState)) => {
      setAnemiaRaw((prev) =>
        typeof updater === "function" ? updater(prev) : { ...prev, ...updater }
      );
    },
    []
  );

  const resetDetect = useCallback(() => setDetectRaw(defaultDetect), []);
  const resetAnemia = useCallback(() => {
    setAnemiaRaw((prev) => {
      // Revoke the old object URL to avoid memory leaks
      if (prev.preview) {
        try { URL.revokeObjectURL(prev.preview); } catch {}
      }
      return defaultAnemia;
    });
  }, []);

  return (
    <AnalysisContext.Provider
      value={{ detect, setDetect, resetDetect, anemia, setAnemia, resetAnemia }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

/* ────────────────────────────────────────────────────────
   Hook — use from any page
   ──────────────────────────────────────────────────────── */

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used inside <AnalysisProvider>");
  return ctx;
}
