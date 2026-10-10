import AlertQueue from "../components/AlertQueue";
import EvaluationPanel from "../components/EvaluationPanel";
import { EdaPanel, ExperimentsPanel } from "../components/EdaPanel";

export default function Dashboard() {
  return (
    <div className="mx-auto flex max-w-[64rem] flex-col gap-14 px-6 pb-24 pt-12 sm:px-10 lg:pt-16">
      <header>
        <p className="text-small font-medium text-accent-ink">FraudLens</p>
        <h1 className="mt-2 text-display">Risk operations cockpit</h1>
        <p className="mt-2 max-w-[56ch] text-body text-ink-2">Ranked review prioritization, not an automatic fraud decision.</p>
      </header>
      <EvaluationPanel />
      <AlertQueue />
      <EdaPanel />
      <ExperimentsPanel />
    </div>
  );
}
