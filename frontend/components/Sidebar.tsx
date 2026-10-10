"use client";

import { useEffect, useState } from "react";

const GROUPS = [
  { label: "Model", items: [
    { id: "overview", label: "Overview" },
    { id: "baselines", label: "Baseline comparison" },
    { id: "pr-curve", label: "Precision-recall" },
    { id: "confusion-matrix", label: "Confusion matrix" },
  ] },
  { label: "Workflow", items: [{ id: "alert-queue", label: "Alert queue" }] },
  { label: "Data", items: [
    { id: "data-exploration", label: "Data exploration" },
    { id: "experiments", label: "Experiments" },
  ] },
];

export default function Sidebar() {
  const [active, setActive] = useState("overview");

  useEffect(() => {
    const ids = GROUPS.flatMap((group) => group.items.map((item) => item.id));
    const update = () => {
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      const present = ids.filter((id) => document.getElementById(id));
      if (atBottom && present.length) return setActive(present[present.length - 1]);
      let current = present[0] ?? "overview";
      for (const id of present) {
        if (document.getElementById(id)!.getBoundingClientRect().top <= 140) current = id;
      }
      setActive(current);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    const observer = new MutationObserver(update);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
      observer.disconnect();
    };
  }, []);

  return (
    <aside className="border-b border-line bg-surface lg:fixed lg:inset-y-0 lg:left-0 lg:z-20 lg:w-60 lg:border-b-0 lg:border-r">
      <div className="flex items-center gap-2.5 px-6 py-5 lg:py-7">
        <span aria-hidden className="grid h-6 w-6 place-items-center rounded-md border border-line-strong bg-raised font-mono text-caption text-accent-ink">F</span>
        <span className="text-body font-semibold tracking-tight">FraudLens</span>
      </div>
      <nav aria-label="Sections" className="flex gap-1 overflow-x-auto px-4 pb-3 lg:flex-col lg:gap-6 lg:overflow-visible lg:pb-0">
        {GROUPS.map((group) => (
          <div key={group.label} className="flex items-center gap-1 lg:flex-col lg:items-stretch">
            <p className="hidden px-2 pb-1 text-caption font-medium uppercase tracking-wider text-ink-3 lg:block">{group.label}</p>
            {group.items.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                aria-current={active === item.id ? "location" : undefined}
                className={`whitespace-nowrap rounded-md px-2 py-1.5 text-small transition-colors active:bg-line ${
                  active === item.id ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised/60 hover:text-ink"
                }`}
              >
                {item.label}
              </a>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
