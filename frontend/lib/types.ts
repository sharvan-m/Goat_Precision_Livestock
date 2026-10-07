export type Health = {
  status: string;
  compute: string;
  weight_model_loaded: boolean;
  weight_model_name: string | null;
  records: number;
  bcs_available: boolean;
};

export type Box = { x1: number; y1: number; x2: number; y2: number };

export type AnalysisResult = {
  goat_id: string;
  detector_confidence: number | null;
  view_score: number | null;
  view_category: string;
  side_class?: string | null;
  side_confidence?: number | null;
  weight_kg: number | null;
  confidence: string | null;
  can_estimate_weight: boolean;
  rejection_reasons: string[];
  measurements: Record<string, number> | null;
  box?: Box;
};

export type Analytics = {
  records: number;
  unique_goats: number;
  average_weight_kg: number | null;
  median_weight_kg: number | null;
  view_breakdown: Record<string, number>;
  weights: number[];
};
