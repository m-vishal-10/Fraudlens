"use client";

import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download } from "lucide-react";
import { API, Evaluation, getJson, pct } from "../lib/api";
import { LoadingBlock, Notice, Section, tableClasses as t } from "./ui";

const COLORS: Record<string, string> = { logistic_regression: "#7C828F", xgboost: "#5B93F0" };
const AXIS = { stroke: "#282C35", tick: { fill: "#9BA0AB", fontSize: 12 } };

function Metric({ label, value, hint, primary }: { label: string; value: string; hint?: string; primary?: boolean }) {
  return (
    <div className="bg-surface p-5">
      <p className="text-caption text-ink-2">{label}</p>
      <p className={`mt-2 font-mono text-metric tabular-nums ${primary ? "text-accent-ink" : "text-ink"}`}>{value}</p>
      {hint && <p className="mt-1.5 text-caption text-ink-3">{hint}</p>}
    </div>
  );
}

export default function EvaluationPanel() {
  const [data, setData] = useState<Evaluation | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJson<Evaluation>("/api/v1/evaluation").then(setData).catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) return <Section id="overview" title="Model performance"><Notice>{error}</Notice></Section>;
  if (!data) {
    return (
      <Section id="overview" title="Model performance">
        <LoadingBlock label="Loading verified evaluation results..." rows={5} />
      </Section>
    );
  }

  const selected = data.models[data.selected_model];
  const m = selected.metrics;
  const names = Object.keys(data.models);
  const curves = names.map((name) => ({
    name,
    points: data.models[name].pr_curve.recall.map((recall, index) => ({ recall, precision: data.models[name].pr_curve.precision[index] })),
  }));

  return (
    <>
      <Section id="overview" title="Overview: model performance" badge="Held-out test set (labeled, retrospective)">
        <p className="mb-6 max-w-[68ch] text-body text-ink-2">
          {data.rows.toLocaleString()} test transactions, {data.fraud_count} actual frauds ({pct(data.fraud_prevalence, 3)} prevalence). Selected model: <strong className="font-medium text-ink">{data.selected_model}</strong>, chosen on validation data. This is a benchmark on a random split, not a live batch.
        </p>
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line lg:grid-cols-5">
          <Metric label="Average Precision (AP)" value={m.average_precision.toFixed(4)} hint="Primary ranking metric" primary />
          <Metric label="ROC-AUC" value={m.roc_auc.toFixed(4)} hint="Secondary; flattering at this imbalance" />
          <Metric label="True fraud in top 200" value={`${m.true_positives} / ${m.k}`} hint="Actual frauds among reviewed alerts" />
          <Metric label="Precision@200" value={pct(m.precision_at_k)} hint="Share of alerts that were fraud" />
          <div className="col-span-2 lg:col-span-1"><Metric label="Recall@200" value={pct(m.recall_at_k)} hint="Share of all fraud captured" /></div>
        </div>
        <p className="mt-6 max-w-[68ch] border-l-2 border-accent pl-4 text-body">{data.selected_queue_statement}</p>
        <a className="mt-5 inline-flex items-center gap-2 rounded text-small text-accent-ink transition-colors hover:text-ink active:text-ink-2" href={`${API}/api/v1/artifacts/top_200_test_queue.csv`} download>
          <Download size={14} aria-hidden /> Download the retrospective top-200 test queue (CSV, includes the true label column)
        </a>
      </Section>

      <Section id="baselines" title="Baseline comparison" badge="Held-out test set">
        <div className={t.wrap}>
          <table className={t.table}>
            <thead>
              <tr>
                <th className={t.th}>Model</th><th className={t.thNum}>Accuracy*</th><th className={t.thNum}>AP</th><th className={t.thNum}>PR-AUC (trapezoid)</th><th className={t.thNum}>ROC-AUC</th><th className={t.thNum}>TP@200</th><th className={t.thNum}>Precision@200</th><th className={t.thNum}>Recall@200</th><th className={t.thNum}>Validation AP</th>
              </tr>
            </thead>
            <tbody>
              <tr className={t.row}>
                <td className={`${t.td} text-ink-2`}>All-negative dummy</td><td className={t.tdNum}>{pct(data.dummy_baseline.accuracy, 2)}</td><td className={`${t.tdNum} text-ink-3`}>n/a</td><td className={`${t.tdNum} text-ink-3`}>n/a</td><td className={`${t.tdNum} text-ink-3`}>n/a</td><td className={t.tdNum}>0</td><td className={t.tdNum}>0.0%</td><td className={t.tdNum}>{pct(data.dummy_baseline.recall)}</td><td className={`${t.tdNum} text-ink-3`}>n/a</td>
              </tr>
              {names.map((name) => {
                const x = data.models[name].metrics;
                const isSelected = name === data.selected_model;
                return (
                  <tr key={name} className={t.row}>
                    <td className={t.td}>
                      <span className="inline-flex items-center gap-2">
                        <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: COLORS[name] }} />
                        {name}
                        {isSelected && <span className="rounded border border-accent/40 px-1.5 text-caption text-accent-ink">selected</span>}
                      </span>
                    </td>
                    <td className={`${t.tdNum} text-ink-3`}>-</td><td className={t.tdNum}>{x.average_precision.toFixed(4)}</td><td className={t.tdNum}>{data.models[name].pr_curve.pr_auc_trapezoidal.toFixed(4)}</td><td className={t.tdNum}>{x.roc_auc.toFixed(4)}</td><td className={t.tdNum}>{x.true_positives}</td><td className={t.tdNum}>{pct(x.precision_at_k)}</td><td className={t.tdNum}>{pct(x.recall_at_k)}</td><td className={t.tdNum}>{data.validation_metrics?.[name]?.average_precision.toFixed(4) ?? "n/a"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-4 max-w-[68ch] text-small text-ink-2">
          *Why accuracy is misleading: predicting every transaction as legitimate scores {pct(data.dummy_baseline.accuracy, 2)} accuracy yet catches 0 of {data.fraud_count} frauds (0% recall). That is why accuracy is not a headline metric here. AP is scikit-learn&apos;s average_precision_score; the trapezoidal PR-AUC column is a different, related number.
        </p>
      </Section>

      <Section id="pr-curve" title="Precision-recall curve" badge="Held-out test set">
        <div className="h-96 rounded-lg border border-line bg-surface p-4 pt-5">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 4, right: 16, bottom: 20, left: 4 }}>
              <CartesianGrid stroke="#1C1F26" vertical={false} />
              <XAxis type="number" dataKey="recall" domain={[0, 1]} {...AXIS} label={{ value: "Recall", position: "insideBottom", offset: -12, fill: "#9BA0AB", fontSize: 12 }} />
              <YAxis type="number" domain={[0, 1]} {...AXIS} label={{ value: "Precision", angle: -90, position: "insideLeft", fill: "#9BA0AB", fontSize: 12 }} />
              <Tooltip formatter={(value: number) => value.toFixed(3)} contentStyle={{ background: "#15181E", border: "1px solid #282C35", borderRadius: 6, fontSize: 12 }} labelStyle={{ color: "#9BA0AB" }} />
              <Legend verticalAlign="top" height={32} wrapperStyle={{ fontSize: 12, color: "#9BA0AB" }} />
              <ReferenceLine y={data.fraud_prevalence} stroke="#6B7080" strokeDasharray="4 4" label={{ value: `Prevalence baseline ${pct(data.fraud_prevalence, 3)}`, fill: "#6B7080", fontSize: 12, position: "insideTopRight" }} />
              {curves.map((curve) => (
                <Line key={curve.name} name={`${curve.name} (AP ${data.models[curve.name].metrics.average_precision.toFixed(3)})`} data={curve.points} dataKey="precision" stroke={COLORS[curve.name]} dot={false} strokeWidth={1.75} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section id="confusion-matrix" title="Confusion matrix" badge={`Held-out test set, ${data.selected_model}, threshold ${data.threshold_for_confusion_matrix}`}>
        <div className={`${t.wrap} inline-block max-w-full`}>
          <table className="text-small">
            <thead>
              <tr><th className="px-5 py-3" /><th className={`${t.th} px-5 text-right`}>Predicted legitimate</th><th className={`${t.th} px-5 text-right`}>Predicted fraud</th></tr>
            </thead>
            <tbody>
              {["Actual legitimate", "Actual fraud"].map((label, row) => (
                <tr key={label} className="border-t border-line">
                  <td className="whitespace-nowrap px-5 py-4 text-ink-2">{label}</td>
                  {m.confusion_matrix[row].map((value, column) => {
                    const correct = row === column;
                    return <td key={column} className={`px-5 py-4 text-right font-mono text-metric tabular-nums ${correct ? "text-ink" : "text-risk-high"}`}>{value.toLocaleString()}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 max-w-[68ch] text-small text-ink-2">The 0.5 threshold is shown only for reference. The operational policy is the rank-based top-200 queue, not a fixed threshold. Scores are not calibrated probabilities.</p>
      </Section>
    </>
  );
}
