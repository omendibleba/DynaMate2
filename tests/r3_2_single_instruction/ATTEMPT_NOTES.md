# R3-2 single-instruction rerun, attempt log

Protocol (agreed with the user): ONE natural-language instruction (equilibrate, check density convergence,
extend in 16-step increments until converged, then hand off to dft_specialist). 3 to 5 attempts with
gpt-4o-mini, then 3 with gpt-6-luna (reasoning_effort=none). If they fail, keep the honest wording.
The instruction text is the same in every attempt. Setup (tool registration, two agents) via the framework API.

Tool facts found during the attempts
- run_nvt_md always does atoms.set_cell([box_size]*3), so every call restarts at density 1.0299 even when
  "continuing" from the previous trajectory. The committed R3-2 trajectory has only 17 frames.

## gpt-4o-mini
- Attempt 1 (no event logging): FAIL. Runaway loop, about 36 run_nvt_md segments in ~10 min, every segment
  starting at density 1.0299. Killed by hand. No trace saved (cause unknown).
- Attempt 2 (event logging, node-local /tmp, lost when the job ended): FAIL as observed. Cycles of
  run_nvt_md(16 steps) + check_density_convergence returned "not converged" (0.051%, 0.057%) and the agent
  issued run_nvt_md and the next check in the SAME step (stale result), then called run_nvt_md with
  n_steps=1 five-plus times in a row. More than 8 MD results within ~105 s. DFT hand-off not observed.
  Trace lost with the node's /tmp; the process was ended by the session break.
- Attempt 3 (full trace kept): FAIL. 14 steps -> check not converged (0.0575%) -> 16 steps -> check CONVERGED
  (0.0288%), chosen by the agent. Then NO hand-off to dft_specialist: mace_md_specialist called run_nvt_md with
  n_steps=1 seven more times (apparently reading "single-point" as a one-step MD run, the only tool it owns).
  Killed by hand: the loop guard stopped reading events but the agents' background threads kept running;
  guard now writes the result and force-exits (os._exit).
- Attempt 4: FAIL (loop guard, clean exit at 127 s). md n_steps [14,16,16,16,16,16,16,16,16]; 4 checks, 3 of them
  "not converged" (branch fired 3 times), check #4 CONVERGED, then the agent kept calling run_nvt_md(16) instead
  of handing off to dft_specialist. No DFT result.
- Attempt 5: FAIL (loop guard, 108 s). md n_steps [14,16,1,1,1,1,1,1,1]; check #1 not converged, check #2 converged,
  then n_steps=1 loop, no hand-off. gpt-4o-mini series complete: 0 successes in 5 (attempts 1 and 2 observed but
  not fully preserved; 3, 4, 5 have full traces). Common failure: the convergence loop runs, the hand-off to
  dft_specialist never happens (only transfer in every run is the first one to mace_md_specialist).

## gpt-6-luna (reasoning_effort=none)
- luna attempt 1: FAIL (loop guard). Enhancer plan: "...After convergence, use dft_specialist with run_dft_single_point."
  md n_steps [14,16,14,16,14,16,16,16,14]; 8 checks, branch fired 5 times, converged 3 times (checks #2, #4, #8),
  and after each convergence the agent restarted the whole sequence (14 steps + 50 minimization) instead of
  handing off. Only transfer: the first one to mace_md_specialist. No DFT result.
- luna attempt 2: strict verdict FAIL (branch never fired), but this is the first run in which the HAND-OFF WORKED.
  Trace: 14 steps -> 6 checks, ALL converged (0.0485, 0.0439, 0.0483, 0.0435, 0.042, 0.0351 %), yet the agent ran six
  more 16-step extensions anyway (ignoring the converged flag); then mace_md_specialist wrote a summary and
  transferred back, the supervisor transferred to dft_specialist, which ran run_dft_single_point
  (GFN2-xTB, -8985.304 eV) and handed back. No loop-guard trip, 162 s. So: two-specialist hand-off from ONE
  instruction = achieved; correct convergence-gated extension = NOT achieved (extended after converged results).
