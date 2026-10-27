"""Workers start from the best program and rotate four research directions."""

from dataclasses import dataclass

WORKERS = 8

REQUESTS = (
    "Refine the current method with one concrete improvement.",
    "Try a different algorithm or representation from the starting solver.",
    "Simplify the method so more useful work fits within the same runtime limit.",
    "Identify an assumption limiting the current method and test a different choice.",
)


@dataclass(frozen=True)
class Experiment:
    attempt: str
    agent: int
    score: float | None
    valid: bool


class Team:
    def __init__(self, baseline, workers=WORKERS, replicate=0):
        self.replicate = replicate
        self.incumbents = [baseline] * workers
        self.best = baseline
        self.counts = [0] * workers

    def select(self, agent):
        if self.best.score > self.incumbents[agent].score + 1e-9:
            self.incumbents[agent] = self.best
        variant = (agent + self.replicate + self.counts[agent]) % len(REQUESTS)
        self.counts[agent] += 1
        return {"agent": agent, "parent": self.best.attempt, "request_variant": variant}

    def observe(self, experiment):
        keep = (
            experiment.valid and experiment.score > self.incumbents[experiment.agent].score + 1e-9
        )
        global_keep = experiment.valid and experiment.score > self.best.score + 1e-9
        if keep:
            self.incumbents[experiment.agent] = experiment
        if global_keep:
            self.best = experiment
        return {
            "decision": "keep" if keep else "discard" if experiment.valid else "invalid",
            "global_improvement": global_keep,
            "global_best": self.best.attempt,
        }


def configuration(workers, replicate):
    return {"agents": workers, "replicate": replicate, "requests": list(REQUESTS)}
