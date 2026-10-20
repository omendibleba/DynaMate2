"""Judge a trial from its saved event list only. Used by trial.py and for post-hoc evaluation of killed runs."""
import ast, json, os, re

TOOLS = ("run_nvt_md", "check_density_convergence", "run_dft_single_point")


def parse(content):
    content = re.sub(r"np\.(?:float64|int64)\(([^)]*)\)", r"\1", content)
    content = content.replace("np.False_", "False").replace("np.True_", "True").replace("np.bool_(False)", "False").replace("np.bool_(True)", "True")
    try:
        return ast.literal_eval(content)
    except Exception:
        try:
            return json.loads(content)
        except Exception:
            return None


def evaluate(trace, model, run, error=None, elapsed=None):
    dft_json = os.path.join(run, "out", "dft_single_point.json")
    events = [(m["name"], m["content"]) for m in trace if m["type"] == "ToolMessage" and m["name"] in TOOLS]
    calls = [c for m in trace for c in m["tool_calls"] if c["name"] in TOOLS]
    md_steps = [c["args"].get("n_steps") for c in calls if c["name"] == "run_nvt_md"]
    checks = [parse(c) for n, c in events if n == "check_density_convergence"]
    checks = [c for c in checks if isinstance(c, dict)]
    seq = [n for n, _ in events]
    reasons = []
    if error:
        reasons.append(error[:140])
    if not checks:
        reasons.append("check_density_convergence never returned a result")
    chk_i = 0
    for idx, (n, c) in enumerate(events):
        if n != "check_density_convergence":
            continue
        res = parse(c)
        nxt = next((x for x, _ in events[idx + 1:] if x != "check_density_convergence"), None)
        if isinstance(res, dict):
            if not res.get("converged") and nxt != "run_nvt_md":
                reasons.append(f"check #{chk_i + 1} not converged but next tool was {nxt}")
            if res.get("converged") and nxt != "run_dft_single_point":
                reasons.append(f"check #{chk_i + 1} converged but next tool was {nxt}")
        chk_i += 1
    if "run_dft_single_point" in seq and not any(c.get("converged") for c in checks):
        reasons.append("DFT ran before any converged check")
    dft_out = json.load(open(dft_json)) if os.path.exists(dft_json) else None
    if not dft_out:
        reasons.append("no DFT result file")
    fired = sum(1 for c in checks if not c.get("converged"))
    if checks and fired == 0:
        reasons.append("converged at first check, so the branch did not fire")
    if checks and not checks[-1].get("converged"):
        reasons.append("last check not converged")
    bad = [s for s in md_steps if s not in (14, 16)]
    if bad:
        reasons.append(f"unexpected n_steps in run_nvt_md calls: {bad}")
    handoff = any(m["type"] == "ToolMessage" and m["name"] == "transfer_to_dft_specialist" for m in trace)
    return {
        "model": model, "outcome": "SUCCESS" if not reasons else "FAIL", "reasons": reasons,
        "elapsed_s": elapsed, "md_n_steps_per_call": md_steps, "n_checks": len(checks), "branch_fired": fired,
        "checks": checks, "handoff_to_dft_specialist": handoff, "dft_energy_eV": (dft_out or {}).get("energy_eV"),
        "tool_sequence": seq, "error": error,
    }
