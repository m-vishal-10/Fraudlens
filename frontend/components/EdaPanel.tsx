"use client";

import { useEffect, useState } from "react";
import { API, EdaSummary, ExperimentRun, getJson, pct } from "../lib/api";
import { LoadingBlock, Notice, Section, Skeleton, tableClasses as t } from "./ui";

const PLOT_TITLES: Record<string, string> = {
  "eda_class_balance.png": "Class balance",
  "eda_amount_by_class.png": "Amount by class",
  "eda_time_by_class.png": "Elapsed time by class",
  "eda_key_features.png": "V14, V17, V12, V10 by class",
  "eda_class_correlations.png": "Correlation with fraud label",
  "eda_feature_correlation.png": "Correlation among top features",
};

const card = "rounded-lg border border-line bg-surface p-5 text-small text-ink-2";

export function EdaPanel() {
  const [data, setData] = useState<EdaSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJson<EdaSummary>("/api/v1/eda/summary").then(setData).catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <Section id="data-exploration" title="Data exploration"><Notice>{error}</Notice></Section>;
  if (!data) {
    return (
      <Section id="data-exploration" title="Data exploration">
        <div role="status" className="space-y-3">
          <span className="sr-only">Loading computed EDA results...</span>
          <Skeleton className="h-4 w-3/4" /><Skeleton className="h-4 w-2/3" /><Skeleton className="h-4 w-1/2" />
          <div className="grid gap-4 pt-3 lg:grid-cols-2"><Skeleton className="h-28" /><Skeleton className="h-28" /></div>
        </div>
      </Section>
    );
  }

  const rates = data.time_bins_2h_elapsed.map((bin) => bin.fraud_rate);
  return (
    <Section id="data-exploration" title="Data exploration" badge="Computed from the full supplied dataset">
      <ul className="grid gap-x-10 gap-y-3 text-body text-ink-2 sm:grid-cols-2">
        <li><strong className="font-medium text-ink">{data.rows.toLocaleString()}</strong> transactions; <strong className="font-medium text-ink">{data.fraud_count}</strong> fraud ({pct(data.fraud_prevalence, 3)}), {data.legitimate_count.toLocaleString()} legitimate</li>
        <li>Missing values: <strong className="font-medium text-ink">{data.missing_values}</strong>; exact duplicate rows: <strong className="font-medium text-ink">{data.duplicate_rows.toLocaleString()}</strong> (reported, not removed)</li>
        <li>All-negative accuracy: <strong className="font-medium text-ink">{pct(data.all_negative_accuracy, 2)}</strong> with 0% fraud recall</li>
        <li>V1-V28 are anonymized PCA components; no real-world meaning is assumed</li>
      </ul>

      <div className="mt-8 grid gap-4 lg:grid-cols-2">
        <div className={card}>
          <p className="text-body font-medium text-ink">Hypothesis: fraud amounts are smaller</p>
          <p className="mt-2">{data.amount_hypothesis.finding}</p>
          <p className="mt-2 font-mono text-caption text-ink-3">Mann-Whitney p-value: {data.amount_hypothesis.mann_whitney_p_value.toExponential(2)}</p>
        </div>
        <div className={card}>
          <p className="text-body font-medium text-ink">Hypothesis: V14, V17, V12, V10 separate the classes</p>
          <ul className="mt-2 space-y-1.5">
            {Object.entries(data.key_features).map(([name, stats]) => (
              <li key={name}><span className="font-mono text-ink">{name}</span>: median fraud <span className="font-mono">{stats.fraud.median.toFixed(2)}</span> vs legitimate <span className="font-mono">{stats.legitimate.median.toFixed(2)}</span>; separation rank <span className="font-mono">{stats.separation_rank_of_30}</span> of 30 features</li>
            ))}
          </ul>
        </div>
        <div className={`${card} lg:col-span-2`}>
          <p className="text-body font-medium text-ink">Time</p>
          <p className="mt-2 max-w-[80ch]">{data.time_note} Transaction counts in the first 24 elapsed hours correlate {data.time_first_vs_second_24h_count_correlation?.toFixed(2) ?? "n/a"} with the next 24, so volume repeats; the 2-hour fraud rate ranges from {pct(Math.min(...rates), 2)} to {pct(Math.max(...rates), 2)} (small counts make single bins noisy). No clock times or calendar dates are implied.</p>
        </div>
      </div>

      <div className="mt-10 grid gap-8 lg:grid-cols-2">
        {data.plots.map((name) => (
          <figure key={name}>
            {/* Backend plots are rendered on white; the frame keeps them legible on the dark canvas. */}
            <div className="overflow-hidden rounded-lg border border-line bg-white p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`${API}/api/v1/artifacts/${name}`} alt={PLOT_TITLES[name] ?? name} loading="lazy" className="w-full" />
            </div>
            <figcaption className="mt-2 text-caption text-ink-2">{PLOT_TITLES[name] ?? name}</figcaption>
          </figure>
        ))}
      </div>
    </Section>
  );
}

export function ExperimentsPanel() {
  const [runs, setRuns] = useState<ExperimentRun[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJson<{ runs: ExperimentRun[] }>("/api/v1/experiments").then((payload) => setRuns(payload.runs)).catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <Section id="experiments" title="Experiment tracking"><Notice>{error}</Notice></Section>;
  if (!runs) return <Section id="experiments" title="Experiment tracking"><LoadingBlock label="Loading MLflow runs..." rows={3} /></Section>;
  if (runs.length === 0) {
    return (
      <Section id="experiments" title="Experiment tracking">
        <div className="rounded-lg border border-dashed border-line-strong p-8 text-center text-small text-ink-2">No runs recorded yet. Run the training script.</div>
      </Section>
    );
  }

  const value = (run: ExperimentRun, suffix: string) => {
    const key = Object.keys(run.metrics).find((name) => name.endsWith(suffix));
    return key ? run.metrics[key].toFixed(4) : "n/a";
  };
  return (
    <Section id="experiments" title="Experiment tracking" badge="MLflow experiment: FraudLens">
      <div className={t.wrap}>
        <table className={t.table}>
          <thead><tr><th className={t.th}>Run</th><th className={t.th}>Role</th><th className={t.thNum}>Validation AP</th><th className={t.thNum}>Test AP</th><th className={t.thNum}>Test ROC-AUC</th><th className={t.thNum}>Run ID</th></tr></thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id} className={t.row}>
                <td className={t.td}>{run.run_name}{run.selected ? " (selected)" : ""}</td><td className={`${t.td} text-ink-2`}>{run.role}</td>
                <td className={t.tdNum}>{value(run, "validation_average_precision")}</td><td className={t.tdNum}>{value(run, "test_average_precision")}</td><td className={t.tdNum}>{value(run, "test_roc_auc")}</td>
                <td className={`${t.tdNum} text-caption text-ink-3`}>{run.run_id.slice(0, 8)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 max-w-[68ch] text-small text-ink-2">Candidate runs contain validation metrics only; held-out test metrics are logged once, in the selected run. Start the MLflow UI with <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-caption text-ink">mlflow ui --backend-store-uri ./mlruns</code> for full run details.</p>
    </Section>
  );
}
