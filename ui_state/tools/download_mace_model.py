def download_mace_model(model_name: str,
                        output_dir: str = '.',
                        model_dict: dict = None,
                        convert_lmp: bool = True):
    """
    Download a MACE foundation model by name from the ACEsuit repositories.

    If the model file already exists in output_dir it will not be downloaded again.
    With convert_lmp=True the model is also converted to LAMMPS format and the
    .pt path is returned; otherwise the raw .model path is returned.

    Parameters
    ----------
    model_name : str  -- canonical model identifier (key in model_dict)
    output_dir : str  -- directory where the file will be saved
    model_dict : dict -- mapping model_name -> URL (uses built-in dict if None)
    convert_lmp : bool -- convert to LAMMPS format after download

    Returns
    -------
    str -- path to the downloaded (or existing) model file
    """
    import os, subprocess

    if model_dict is None:
        model_dict = {
            "MACE-MP-0b":  "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mace_agnesi_medium.model",
            "MACE-MP-0b2": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model",
            "MACE-MP-0b3": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
            "MACE-MPA-0":  "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
            "MACE-OMAT-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
            "MACE-MATPES-PBE-0":    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
            "MACE-MATPES-r2SCAN-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
            "MACE-MH-0":   "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
        }

    if model_name not in model_dict:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list(model_dict.keys())}"
        )

    url      = model_dict[model_name]
    fname    = os.path.basename(url)
    out_path = os.path.join(output_dir, fname)

    if os.path.exists(out_path):
        print(f"Model '{model_name}' already exists at {out_path}. Skipping download.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading '{model_name}' from:\n{url}\n-> {out_path}")
        subprocess.run(["wget", "-O", out_path, url], check=True)
        print(f"Download complete: {out_path}")

    if convert_lmp:
        print(f"Converting '{model_name}' to LAMMPS format.")
        os.system(f"python ~/mace/mace/cli/create_lammps_model.py {out_path}")
        return out_path + '-lammps.pt'
    return out_path
