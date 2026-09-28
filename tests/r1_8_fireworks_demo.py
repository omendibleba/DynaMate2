"""
R1-8 demonstration: DynaMate2's fireworks_dispatch_tools.py dispatching a real
temperature-sweep workflow to an actual FireWorks LaunchPad (local MongoDB),
then actually executing it through FireWorks' own worker (rapidfire), not
just submitting and reading back a READY state.

This is the "how FireWorks can be implemented in the DynaMate2 framework"
evidence for the R1-8 response -- a DynaMate2 tool calling LaunchPad.add_wf()
with real fw_ids/status, not a head-to-head benchmark.

As run for the manuscript, this executed inside the dynamate2_gpu Apptainer
container (--bind <repo>:/app), against a local, no-root MongoDB (the
portable mongodb-linux-x86_64-rhel90 tarball, `mongod --dbpath ... --port
27117 --bind_ip 127.0.0.1`), with `fireworks`/`pymongo` pip-installed to a
separate --target directory (the container filesystem is read-only and sets
PYTHONNOUSERSITE=1, so plain `pip install --user` won't work) added to
PYTHONPATH. A my_launchpad.yaml (host/port/name pointing at that mongod)
must exist in the working directory this script runs from, since
check_fireworks_status() uses FireWorks' own LaunchPad.auto_load() config
search rather than taking an explicit launchpad_file argument. See
tests/r1_8_fireworks_results.json for the exact output this produced.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for tutorials.*

from fireworks import LaunchPad
from fireworks.core.fworker import FWorker
from fireworks.core.rocket_launcher import rapidfire

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tutorials"))
from fireworks_dispatch_tools import submit_temperature_sweep_to_fireworks, check_fireworks_status

LAUNCHPAD_FILE = "my_launchpad.yaml"  # must exist in cwd -- see module docstring
STRUCTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tutorials", "sample_outputs", "end_to_end_test", "nacl_water_box.xyz",
)
OUTPUT_DIR = "fireworks_out"

print("=== 0. Initializing a fresh LaunchPad database ===", flush=True)
_lp = LaunchPad.from_file(LAUNCHPAD_FILE)
_lp.reset(None, require_password=False)

print("=== 1. Submitting a 2-temperature NVT sweep to a real FireWorks LaunchPad ===", flush=True)
result = submit_temperature_sweep_to_fireworks(
    structure_file=STRUCTURE,
    model_name="polar-1-m",
    temperatures_K=[290, 310],
    n_steps=20,
    box_size=12.5,
    output_dir=OUTPUT_DIR,
    launchpad_file=LAUNCHPAD_FILE,
)
print(json.dumps(result, indent=2), flush=True)

print("\n=== 2. Status immediately after submission ===", flush=True)
fw_id_list = list(result["fw_ids"].values())
status_before = check_fireworks_status(fw_id_list)
print(json.dumps(status_before, indent=2), flush=True)

print("\n=== 3. Running the workflow through a real FireWorks worker (rapidfire) ===", flush=True)
launchpad = LaunchPad.from_file(LAUNCHPAD_FILE)
rapidfire(launchpad, fworker=FWorker(), nlaunches=0, sleep_time=1)

print("\n=== 4. Status after execution ===", flush=True)
status_after = check_fireworks_status(fw_id_list)
print(json.dumps(status_after, indent=2), flush=True)

with open("/scratchpad/fireworks_demo_results.json", "w") as f:
    json.dump({
        "submission_result": result,
        "status_before": {str(k): v for k, v in status_before.items()},
        "status_after": {str(k): v for k, v in status_after.items()},
    }, f, indent=2)

print("\nSaved to /scratchpad/fireworks_demo_results.json", flush=True)
