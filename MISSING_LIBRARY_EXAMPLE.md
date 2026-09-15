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
note: from another terminal, run 'apptainer exec instance://dynamate2-8888 pip install <package>' to add a library to THIS session.
```

**Why you have to relaunch instead of just installing into the already-running session**:
writability is a property of how the container was started — Apptainer's `--writable-tmpfs`
overlay only exists for a session launched with that flag from the beginning. There's no way
to add it to a session already running without it.

## 4. Install the missing library — from a second terminal, directly

`--writable` launches the session as a named Apptainer instance (`dynamate2-<port>`, printed
in the note above) specifically so a second terminal can install into it directly:

```bash
apptainer exec instance://dynamate2-8888 pip install pymatgen
```

(Adjust the port if you launched with `DYNAMATE_PORT` set to something other than `8888`.)

**Do this instead of asking the chat agent to run the install for you.** Asking the agent
(e.g. "please install the pymatgen package") routes through an LLM translating a
natural-language request into a shell command — tested directly, and it's not reliable: one
real attempt came back "Hello from shell agent!," not real pip output, and nothing had
actually been installed. The command above is deterministic, runs the real `pip install`
yourself, and its output is real, verifiable pip output — not a chat response to trust or
distrust.

You can verify it landed before even touching the UI:
```bash
apptainer exec instance://dynamate2-8888 python3 -c "import pymatgen; print('ok')"
```

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

## How the second-terminal install actually works

Apptainer supports two different ways of running a container:

- **`apptainer run`**: a normal foreground process. Nothing else can attach to it once
  started.
- **`apptainer instance start <sif> <name>`**: a *named*, backgrounded container that other
  `apptainer exec instance://<name> <command>` calls *can* attach to from separate
  terminals, as long as they're run by the same user.

Under `--writable`, `run.sh` uses the second form: it starts the container as a fixed,
predictable instance (`dynamate2-<port>`), then execs the actual app (`python server.py`)
against that instance as the process you watch in your terminal — same experience as
before, just reachable from elsewhere too. This was verified directly against this repo's
own `.sif` files, including that `--nv` (GPU access) set once at `instance start` correctly
carries over to later `exec` calls without needing to repeat it, and that an installed
package genuinely persists across independent `exec` invocations into the same instance.

The non-`--writable` path is unchanged — still a plain `apptainer run`, since there's
nothing to attach to when the session isn't writable anyway.
