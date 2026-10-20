# Reliability of the end-to-end example (Section 5.2)

The tutorial notebook's final integration cell was repeated 10 times from a fresh state (new pool state and output
folder per run) with `gpt-4.1-mini` at temperature 0, the model the notebook uses. The five tools are registered and
assigned to `mace_md_specialist` through the framework API, which is the state the notebook reaches before that cell.
The query is the notebook's own, sent through the PromptEnhancer and the supervisor (`recursion_limit=60`).

**Success criteria (fixed before any run):** all seven output files exist and are non-empty; the box reads as
H128ClNaO64; the trajectory has 101 frames and a final density between 0.95 and 1.10 g/cm3; `smiles_to_xyz` is
called for O, [Na+], and [Cl-]; `packmol_build_system` is called with 64 water, 1 Na, 1 Cl, and a 12.5 A box;
`run_nvt_md` is called exactly once with n_steps=100, pressure_bar=1.0, minimize_steps=50, 300 K, polar-1-m;
`plot_nvt_trajectory` is called; `download_mace_model` is not called; no guard trips (15 tool results, 10 minutes).

**Result:** 10 of 10 runs met every criterion (Wilson 95% interval for the success rate: 72% to 100%).
Every run made the same six tool calls in the same order, the box file was byte-identical to the committed one,
the final density was 1.0344 to 1.0369 g/cm3, and each run took 64 to 68 s.

**What this does and does not show.** It covers one fixed instruction, one model, one system, and tools registered
through the API. Temperature 0 and identical tool sequences mean low run-to-run variance, not that failures are
impossible. It says nothing about larger tool sets or iterative control, where the single-instruction attempts in
`tests/r3_2_single_instruction/` show lower reliability.

Run it with `run_all.sh <model> <first> <last>`.
