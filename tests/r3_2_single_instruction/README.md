# R3-2 single-instruction attempts

The convergence-gated MD-to-DFT demonstration in Section 5.4 (`tests/r3_2_md_dft_demo.py`) is driven by a short
script that reads each `check_density_convergence` result and issues the next instruction. These files document
our attempts to have the agents run the whole loop from ONE instruction instead.

**Instruction (identical in every attempt):** run a 14-step NPT segment with `run_nvt_md`, call
`check_density_convergence` (window_frames=5, tolerance_percent=0.05), extend by 16 steps and re-check until it
reports convergence, then use `dft_specialist` for a GFN2-xTB single point. Setup (tool registration, two agents)
goes through the framework API. The PromptEnhancer rewrites the instruction on the model under test, so its added
plan differs slightly between models (gpt-4o-mini: "...and finally use dft_specialist...", gpt-6-luna: "...After
convergence, use dft_specialist...").

**Success criterion (from the saved trace only):** at least one not-converged check followed by an extension,
no extension after a converged check, a hand-off to `dft_specialist` only after a converged check, and a DFT
result file. `evaluate.py` applies it. A guard stops a run after 8 `run_nvt_md` results or 25 minutes.

| Series | Attempts | Convergence loop ran | Hand-off to dft_specialist | Stopped when check said converged |
|---|---|---|---|---|
| gpt-4o-mini | 5 | 5 of 5 | 0 of 5 | no, kept calling `run_nvt_md` |
| gpt-6-luna (reasoning_effort=none) | 3 | 3 of 3 | 2 of 3 | no, extended or restarted after "converged" |

By the strict criterion all eight attempts failed. Full event traces are in `results/` for gpt-4o-mini attempts 3 to
5 and gpt-6-luna attempts 1 to 3. gpt-4o-mini attempts 1 and 2 were observed live (runaway loops of `run_nvt_md`,
no hand-off) but their traces were not preserved, so they are described in `ATTEMPT_NOTES.md` only.

**Tool behavior worth knowing:** `run_nvt_md` resets the cell to `box_size` on every call, so a "continued"
segment restarts at the initial density. The 46 steps reported in Section 5.4 are therefore the sum of separate
segments, not one continuous equilibration.

`run_trial.sh` shows how to run one attempt (path names inside `trial.py` use the container binds `/app` and
`/mnt/work`).
