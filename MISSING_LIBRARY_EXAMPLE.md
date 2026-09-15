# Worked Example: Bringing a Function That Needs a Library Not in the Image

A complete walkthrough, start to finish, for the situation this is actually for: a
tutorial/workshop participant brings their own function, and it needs a Python library
DynaMate2's container doesn't have. See [`--writable`](README.md#run-dynamate2) in the
main README for the short version — this is the same thing, worked through in full.

**Scenario**: a participant wants to register a custom tool that parses crystal structure
files using [`pymatgen`](https://pymatgen.org/) — not installed in the DynaMate2 image.

---

## 1. Clone and launch (normal Quick Start flow)

```bash
git clone https://github.com/omendibleba/DynaMate2.git
cd DynaMate2
cp .env_sample .env   # fill in OPENAI_API_KEY
mkdir -p containers
ln -sf /groups/ycolon/Orlando/containers/dynamate2_gpu.sif containers/dynamate2_gpu.sif
./run.sh --gpu
```

## 2. Register the function, then hit the error when actually using it

In the chat UI:

> "Here's a function I'd like to add — it reads a CIF file and returns the space group:
> ```python
> def get_space_group(cif_path: str) -> str:
>     from pymatgen.core import Structure
>     from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
>     structure = Structure.from_file(cif_path)
>     return SpacegroupAnalyzer(structure).get_space_group_symbol()
> ```
> Please register it and assign it to compute_agent."

Registration itself succeeds — it just stores the source, it doesn't execute the function
body or check its imports. The error only shows up the first time the function actually
**runs**:

> "Use get_space_group on my_structure.cif"

```
ModuleNotFoundError: No module named 'pymatgen'
```

## 3. Stop and relaunch with `--writable`

Back in the terminal running `run.sh`:
```
Ctrl+C
```
```bash
./run.sh --gpu --writable
```
Confirms it's active:
```
note: --writable is on -- pip installs work this session, but are lost when it ends.
```

**Why you have to relaunch instead of just installing into the already-running session**:
writability is a property of how the container was started — Apptainer's `--writable-tmpfs`
overlay only exists for a session launched with that flag from the beginning. There's also
no supported way to reach into an *already-running* `apptainer run` process from a separate
terminal to install something into it — `apptainer`'s `instance://` addressing only works
for containers started via `apptainer instance start` (a different, named-service launch
mode), which is not what `run.sh` uses. So: stop, relaunch with `--writable`, continue.

## 4. Install the missing library — from inside the same session

Because there's no supported way to attach a second terminal to an already-running
`apptainer run`, the install has to happen **inside the same running session** that
`run.sh --writable` started. The natural way to do that: just ask the agent, which runs
the command via `shell_agent` in that same container process:

> "Please install the pymatgen Python package."

(If you're comfortable with it, you can also literally hand the agent a `pip install`
shell command yourself the same way — same mechanism.)

## 5. Retry — same prompt as step 2

> "Use get_space_group on my_structure.cif"

This time it actually runs, and returns the space group instead of erroring.

## 6. After the tutorial: make it permanent

Nothing installed via `--writable` survives past that session — the next `./run.sh --gpu`
(without `--writable`) is back to the original image, `pymatgen` gone. If the function is
worth keeping around:

1. Add `pymatgen` to `docker/environment.cpu.yml` and/or `docker/environment.gpu.yml` (or a
   dedicated `pip install` line in the `Dockerfile`, matching the pattern already used for
   `graph_electrostatics` — see the comments right above the `mace-torch` install step).
2. Push, wait for CI to publish a new image (`docker-publish.yml`).
3. Rebuild the `.sif`:
   ```bash
   export APPTAINER_CACHEDIR=/groups/ycolon/Orlando/containers/.apptainer-cache
   mkdir -p "$APPTAINER_CACHEDIR"
   apptainer pull -F /groups/ycolon/Orlando/containers/dynamate2_gpu.sif \
     docker://ghcr.io/omendibleba/dynamate2:gpu
   ```

From then on, every `--gpu` session (writable or not) has `pymatgen` built in — no one
needs to repeat steps 3–4 again for this particular library.

---

## Why there's no "just exec into the running container from another terminal" option

Worth understanding if you're troubleshooting mid-tutorial and instinctively reach for a
second terminal: Apptainer supports two different ways of running a container —

- **`apptainer run`** (what `run.sh` uses): a normal foreground process. Nothing else can
  attach to it afterward; the only way in is through commands the process itself runs
  (i.e. through the agent, via `shell_agent`).
- **`apptainer instance start <sif> <name>`**: a *named*, backgrounded container that other
  `apptainer exec instance://<name> ...` commands *can* attach to from separate terminals,
  as long as they're run by the same user.

`run.sh` deliberately uses the first form (`apptainer run`) — it's simpler, and matches how
`docker run` behaves too (one process, one terminal, `Ctrl+C` to stop). Switching to the
instance-based form to support a genuine "install from a second terminal" workflow would be
a real, separate change to how `run.sh` launches things — not something either does today.
If that turns out to actually be needed, it's worth revisiting deliberately rather than
assuming the `instance://` syntax already works (it doesn't, for what `run.sh` starts).
