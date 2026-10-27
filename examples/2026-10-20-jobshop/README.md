# Factory scheduling

<p align="center">
  <img src="../../docs/images/jobshop.png" width="760" alt="Experiment overview">
</p>

Forty jobs compete for twenty machines, with 800 operations to schedule. Flint changes the order of operations; the evaluator checks the schedule and measures when all work finishes. [Task details](../../tasks/jobshop).

Across **100 attempts**, finishing time fell from **3,438 to 2,545**, a **26.0% reduction** from the round-robin baseline. Ten changes were retained. The best appeared at attempt 71; the remaining 29 attempts did not improve it.

The plot highlights [64 randomized greedy restarts](028/solver.py), [operation reinsertion with changed initialization and annealing](034/solver.py), and [machine-workload bounds with moves in both directions](071/solver.py).

There were 89 valid evaluations: 79 distinct programs and 10 repeats. Eleven attempts failed: two invalid schedules, three truncated responses, five CPU-limit exits and one program error. All attempts are included in the [annotated report](report.html), with their prompts, raw streams, programs, measurements and decisions.

The run used one synthetic instance, seed 0, and limits of 15 CPU seconds and 25 wall seconds per candidate. Programs that stop by elapsed time can vary even with a fixed seed. Model responses took 3.21 hours; [full measurements](notes.json) include token counts and evaluation time.

## Timelines and progress

<p align="center">
  <img src="search-progress.png" width="880" alt="Machine timelines and research progress">
</p>

```sh
uv run python -m tools.verify_run examples/2026-10-20-jobshop
```

Rebuild these figures from the saved schedules and response times:

```sh
uv run python -m tools.render_jobshop_progress
```

This reads recorded results without executing candidates or calling the model.
