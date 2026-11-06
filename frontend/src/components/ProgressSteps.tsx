import { PIPELINE_STEPS, type Song, type SongStatus } from "../api/types";

type StepState = "done" | "current" | "upcoming";

export function ProgressSteps({ song }: { song: Song }) {
  return (
    <ol className="steps">
      {PIPELINE_STEPS.map((step) => {
        const state = stepState(step.status, song.status);
        const timing = song.step_timings[step.status];
        return (
          <li key={step.status} className={`step ${state}`}>
            <span className="step-mark" aria-hidden="true" />
            <span>{step.label}</span>
            {state === "done" && timing !== undefined && (
              <span className="step-time">{timing.toFixed(1)}s</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function stepState(step: SongStatus, current: SongStatus): StepState {
  const order = PIPELINE_STEPS.map((item) => item.status);
  const stepIndex = order.indexOf(step);
  const currentIndex = order.indexOf(current);
  if (currentIndex === -1 || stepIndex < currentIndex) {
    return "done";
  }
  if (stepIndex === currentIndex) {
    return "current";
  }
  return "upcoming";
}
