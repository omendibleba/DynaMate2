# Sample Outputs

These files are the **expected outputs** of `DynaMate2_tutorial.ipynb`.

Running the notebook re-generates files in two locations:

| Test | Writes to |
|---|---|
| T2 — DMF box | `tutorials/` root: `dmf.xyz`, `dmf_30.xyz` |
| T2.1 — NaCl-water box | `tutorials/` root: `water.xyz`, `nacl_water_box.xyz` |
| T3 — NVT simulation | `tutorials/` root: `dmf_30.traj` |
| T4 — Trajectory plot | `tutorials/` root: `dmf_30.png` |
| Final integration test | `tutorials/sample_outputs/end_to_end_test/` |

The files in this directory are committed reference copies. Use them to verify
that your run produced the correct outputs (same file sizes and content).

> **Note:** Individual test outputs written to `tutorials/` root are gitignored.
> Only `sample_outputs/` is tracked in git.
