import { AlertTriangle } from "lucide-react";

export function Section({
  id,
  title,
  badge,
  action,
  children,
}: {
  id: string;
  title: string;
  badge?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-10 border-t border-line pt-10">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <h2 id={`${id}-title`} className="text-title">{title}</h2>
          {badge && <span className="rounded border border-line-strong px-2 py-0.5 font-mono text-caption text-ink-2">{badge}</span>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Notice({ children }: { children: React.ReactNode }) {
  return (
    <div role="alert" className="flex gap-3 rounded-lg border border-risk-medium/30 bg-risk-medium/5 p-4 text-small text-ink">
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-risk-medium" aria-hidden />
      <div>{children}</div>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`rounded bg-raised motion-safe:animate-pulse ${className}`} />;
}

export function LoadingBlock({ label, rows = 3 }: { label: string; rows?: number }) {
  return (
    <div role="status" className="space-y-3">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-4 w-2/3" />
      <div className="grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3">
        {Array.from({ length: rows }, (_, index) => (
          <div key={index} className="space-y-3 bg-surface p-5">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-6 w-24" />
          </div>
        ))}
      </div>
    </div>
  );
}

export const tableClasses = {
  wrap: "overflow-x-auto rounded-lg border border-line bg-surface",
  table: "w-full min-w-max text-left text-small",
  th: "whitespace-nowrap px-4 py-2.5 text-caption font-medium text-ink-2",
  thNum: "whitespace-nowrap px-4 py-2.5 text-right text-caption font-medium text-ink-2",
  td: "whitespace-nowrap px-4 py-2.5",
  tdNum: "whitespace-nowrap px-4 py-2.5 text-right font-mono tabular-nums",
  row: "border-t border-line transition-colors hover:bg-raised",
};
