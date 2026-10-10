"use client";

import { useEffect, useRef, useState } from "react";
import { FileUp, Loader2, X } from "lucide-react";
import { API } from "../lib/api";
import { Notice, Section, Skeleton, tableClasses as t } from "./ui";

type Reason = { feature: string; value: number; contribution: number; direction: string; description: string };
export type Alert = { rank: number; transaction_id: string; risk_score: number; amount: number; time: number; risk_category: string; analyst_note: string; reasons: Reason[] };

const CATEGORY_DOT: Record<string, string> = { High: "bg-risk-high", Medium: "bg-risk-medium", Review: "bg-risk-review" };

export default function AlertQueue() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  async function uploadBatch(file: File) {
    setUploading(true);
    setUploadError("");
    setSelected(null);
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API}/api/v1/alerts/upload`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "The batch could not be scored.");
      setAlerts(payload.alerts);
    } catch (reason) {
      setAlerts([]);
      setUploadError(reason instanceof Error ? reason.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  const active = alerts.find((alert) => alert.transaction_id === selected) ?? null;

  return (
    <Section
      id="alert-queue"
      title="Alert queue (uploaded batch, unlabeled)"
      action={
        <label className={`inline-flex items-center gap-2 rounded-md bg-accent px-3.5 py-2 text-small font-medium text-white transition-colors focus-within:ring-2 focus-within:ring-accent/60 focus-within:ring-offset-2 focus-within:ring-offset-canvas ${uploading ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-accent-hover active:bg-accent-press"}`}>
          {uploading ? <Loader2 size={14} className="motion-safe:animate-spin" aria-hidden /> : <FileUp size={14} aria-hidden />}
          {uploading ? "Scoring..." : "Upload feature CSV"}
          <input type="file" accept=".csv,text/csv" className="sr-only" disabled={uploading} onChange={(event) => { const file = event.target.files?.[0]; if (file) uploadBatch(file); event.target.value = ""; }} />
        </label>
      }
    >
      {uploadError && <div className="mb-4"><Notice>{uploadError}</Notice></div>}

      {uploading ? (
        <div role="status" className={t.wrap}>
          <span className="sr-only">Scoring the uploaded batch</span>
          {Array.from({ length: 8 }, (_, index) => (
            <div key={index} className="flex items-center gap-6 border-t border-line px-4 py-3 first:border-t-0">
              <Skeleton className="h-3.5 w-8" /><Skeleton className="h-3.5 w-28" /><Skeleton className="ml-auto h-3.5 w-16" /><Skeleton className="h-3.5 w-16" /><Skeleton className="h-3.5 w-20" />
            </div>
          ))}
        </div>
      ) : alerts.length === 0 ? (
        <div className="rounded-lg border border-dashed border-line-strong px-6 py-14 text-center">
          <FileUp size={20} className="mx-auto text-ink-3" aria-hidden />
          <p className="mx-auto mt-4 max-w-[52ch] text-small text-ink-2">Upload a CSV with Time, V1-V28 and Amount (no Class column). At most 200 alerts are returned, highest risk first.</p>
        </div>
      ) : (
        <div className={`${t.wrap} max-h-[34rem] overflow-y-auto`}>
          <table className={t.table}>
            <thead className="sticky top-0 z-10 bg-surface shadow-[0_1px_0_#1C1F26]">
              <tr><th className={t.thNum}>Rank</th><th className={t.th}>Transaction ID</th><th className={t.thNum}>Risk score</th><th className={t.thNum}>Amount</th><th className={t.th}>Category</th><th className={t.th}>Why flagged</th></tr>
            </thead>
            <tbody>
              {alerts.map((alert) => (
                <Row key={alert.transaction_id} alert={alert} open={selected === alert.transaction_id} onToggle={() => setSelected(selected === alert.transaction_id ? null : alert.transaction_id)} />
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-4 max-w-[68ch] text-caption text-ink-3">Risk scores are uncalibrated model scores that prioritize human review; they do not confirm fraud. Explanations describe model behavior, and V1-V28 are anonymized components.</p>
      {active && <Drawer alert={active} onClose={() => setSelected(null)} />}
    </Section>
  );
}

function Row({ alert, open, onToggle }: { alert: Alert; open: boolean; onToggle: () => void }) {
  const top = alert.reasons.filter((reason) => reason.direction === "increases").slice(0, 3).map((reason) => reason.feature).join(", ");
  return (
    <tr onClick={onToggle} className={`cursor-pointer border-t border-line transition-colors first:border-t-0 ${open ? "bg-raised" : "hover:bg-raised/60"}`}>
      <td className={`${t.tdNum} text-ink-2`}>{alert.rank}</td>
      <td className={`${t.td} font-mono`}>{alert.transaction_id}</td>
      <td className={t.tdNum}>{alert.risk_score.toFixed(4)}</td>
      <td className={t.tdNum}>{alert.amount.toFixed(2)}</td>
      <td className={t.td}><span className="inline-flex items-center gap-2"><span aria-hidden className={`h-1.5 w-1.5 rounded-full ${CATEGORY_DOT[alert.risk_category] ?? "bg-risk-review"}`} />{alert.risk_category}</span></td>
      <td className={t.td}>
        <button type="button" aria-haspopup="dialog" aria-expanded={open} className="rounded border border-line-strong px-2 py-1 font-mono text-caption text-accent-ink transition-colors hover:bg-line active:bg-line-strong">
          {open ? "Hide" : top || "Details"}
        </button>
      </td>
    </tr>
  );
}

function Drawer({ alert, onClose }: { alert: Alert; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const maxContribution = Math.max(...alert.reasons.map((reason) => Math.abs(reason.contribution)), 1e-9);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onCloseRef.current(); };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); previous?.focus(); };
  }, []);

  return (
    <div className="fixed inset-0 z-40">
      <div onClick={onClose} aria-hidden className="absolute inset-0 bg-black/50 motion-safe:animate-fade-in" />
      <aside role="dialog" aria-modal="true" aria-label={`Alert details for ${alert.transaction_id}`} className="absolute inset-y-0 right-0 flex w-full max-w-[28rem] flex-col border-l border-line bg-surface motion-safe:animate-slide-in">
        <header className="flex items-start justify-between gap-4 border-b border-line px-6 py-5">
          <div>
            <p className="text-caption text-ink-2">Priority #{alert.rank}</p>
            <h3 className="mt-1 font-mono text-title">{alert.transaction_id}</h3>
          </div>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close details" className="rounded-md p-1.5 text-ink-2 transition-colors hover:bg-raised hover:text-ink active:bg-line"><X size={16} aria-hidden /></button>
        </header>
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <dl className="grid grid-cols-3 gap-px overflow-hidden rounded-lg border border-line bg-line">
            <div className="bg-surface p-4"><dt className="text-caption text-ink-2">Risk score</dt><dd className="mt-1 font-mono text-body tabular-nums">{alert.risk_score.toFixed(4)}</dd></div>
            <div className="bg-surface p-4"><dt className="text-caption text-ink-2">Amount</dt><dd className="mt-1 font-mono text-body tabular-nums">{alert.amount.toFixed(2)}</dd></div>
            <div className="bg-surface p-4"><dt className="text-caption text-ink-2">Category</dt><dd className="mt-1 inline-flex items-center gap-2 text-body"><span aria-hidden className={`h-1.5 w-1.5 rounded-full ${CATEGORY_DOT[alert.risk_category] ?? "bg-risk-review"}`} />{alert.risk_category}</dd></div>
          </dl>
          <p className="mt-6 text-small text-ink-2">{alert.analyst_note}</p>
          <h4 className="mt-8 text-caption font-medium uppercase tracking-wider text-ink-3">Why flagged</h4>
          <ul className="mt-4 space-y-5">
            {alert.reasons.map((reason) => {
              const up = reason.direction === "increases";
              return (
                <li key={reason.feature}>
                  <div className="flex items-baseline justify-between gap-3 text-small">
                    <span><span className="font-mono text-ink">{reason.feature}</span> <span className="font-mono text-ink-2">= {reason.value.toFixed(3)}</span></span>
                    <span className="text-ink-2">{reason.direction} risk <span className={`font-mono ${up ? "text-risk-high" : "text-risk-ok"}`}>({reason.contribution >= 0 ? "+" : ""}{reason.contribution.toFixed(2)})</span></span>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-raised">
                    <div className={`h-full rounded-full ${up ? "bg-risk-high" : "bg-risk-ok"}`} style={{ width: `${(Math.abs(reason.contribution) / maxContribution) * 100}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      </aside>
    </div>
  );
}
