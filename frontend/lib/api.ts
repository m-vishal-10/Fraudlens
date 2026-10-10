export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Metrics = {
  average_precision: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  k: number;
  true_positives: number;
  precision_at_k: number | null;
  recall_at_k: number | null;
};

export type ModelReport = {
  metrics: Metrics;
  pr_curve: { precision: number[]; recall: number[]; pr_auc_trapezoidal: number };
  queue_statement: string;
};

export type Evaluation = {
  evaluation_split: string;
  note: string;
  rows: number;
  fraud_count: number;
  fraud_prevalence: number;
  threshold_for_confusion_matrix: number;
  selected_model: string;
  selection_basis: string;
  capacity: number;
  dummy_baseline: { accuracy: number; precision: number; recall: number; confusion_matrix: number[][] };
  models: Record<string, ModelReport>;
  selected_queue_statement: string;
  validation_metrics: Record<string, Metrics> | null;
  class_counts: Record<string, { legitimate: number; fraud: number }> | null;
};

export type FeatureStats = { median: number; mean: number };
export type EdaSummary = {
  rows: number;
  fraud_count: number;
  legitimate_count: number;
  fraud_prevalence: number;
  missing_values: number;
  duplicate_rows: number;
  all_negative_accuracy: number;
  amount_hypothesis: { hypothesis: string; finding: string; mann_whitney_p_value: number };
  key_features: Record<string, { fraud: FeatureStats; legitimate: FeatureStats; separation_rank_of_30: number; single_feature_auc: number }>;
  top_correlations_with_class: Record<string, number>;
  time_note: string;
  time_first_vs_second_24h_count_correlation: number | null;
  time_bins_2h_elapsed: { fraud_rate: number }[];
  plots: string[];
};

export type ExperimentRun = {
  run_id: string;
  run_name: string;
  role: string | null;
  selected: boolean;
  status: string;
  metrics: Record<string, number>;
};

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status}).`);
  }
  return payload as T;
}

export const pct = (value: number | null | undefined, digits = 1) =>
  value == null ? "Unavailable" : `${(value * 100).toFixed(digits)}%`;
