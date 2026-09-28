def submit_temperature_sweep_to_fireworks(
    structure_file: str,
    model_name: str,
    temperatures_K: list,
    n_steps: int,
    box_size: float,
    output_dir: str = '.',
    launchpad_file: str = None,
    category: str = None,
) -> dict:
    """
    Submit one independent NVT molecular dynamics run per temperature to a
    FireWorks LaunchPad, reusing the already-registered run_nvt_md function as
    the computational step. Each temperature becomes its own Firework wrapping
    run_nvt_md via FireWorks' PyTask, so no simulation code is duplicated or
    rewritten for the workflow-management layer; the same function that DynaMate2
    calls directly is dispatched, unchanged, to FireWorks for queued execution.

    Parameters
    ----------
    structure_file : str        -- path to the input structure (XYZ or extxyz),
                                    typically the output of packmol_build_system
    model_name     : str        -- name of the MACE polar foundation model to use
                                    (e.g. 'polar-1-m')
    temperatures_K : list[float]-- one NVT run is submitted per entry, e.g. [280, 300, 320]
    n_steps        : int        -- number of MD steps per run (same for every temperature)
    box_size       : float      -- size of the cubic simulation box in Angstroms
    output_dir     : str        -- directory for per-temperature trajectory and log files
    launchpad_file : str        -- path to a FireWorks my_launchpad.yaml; if None,
                                    FireWorks' default LaunchPad.auto_load() is used
    category       : str        -- optional FireWorks worker category, for routing
                                    to a specific queue (e.g. a GPU partition)

    Returns
    -------
    dict -- {
        "fw_ids": {temperature_K: fw_id, ...},
        "wf_id": workflow id returned by the LaunchPad,
        "launchpad": str(launchpad),
    }
    """
    import os

    from fireworks import Firework, LaunchPad, PyTask, Workflow

    if launchpad_file:
        launchpad = LaunchPad.from_file(launchpad_file)
    else:
        launchpad = LaunchPad.auto_load()

    os.makedirs(output_dir, exist_ok=True)

    fireworks = []
    temperature_by_placeholder_id = {}
    for temperature_K in temperatures_K:
        output_traj = os.path.join(output_dir, f'nvt_{temperature_K:g}K.traj')
        log_file = os.path.join(output_dir, f'nvt_{temperature_K:g}K.log')

        task = PyTask(
            func='tutorials.ASE_NVT_PBC.run_nvt_md',
            kwargs={
                'structure_file': structure_file,
                'box_size': box_size,
                'temperature_K': temperature_K,
                'n_steps': n_steps,
                'model_name': model_name,
                'output_traj': output_traj,
                'log_file': log_file,
            },
        )
        spec = {'_category': category} if category else {}
        fw = Firework(task, name=f'nvt_md_{temperature_K:g}K', spec=spec)
        fireworks.append(fw)
        # fw.fw_id is a locally-assigned placeholder id until the workflow is
        # added to the LaunchPad; add_wf's return value maps each placeholder
        # id to the real, LaunchPad-assigned id.
        temperature_by_placeholder_id[fw.fw_id] = temperature_K

    workflow = Workflow(fireworks, name='dynamate2_temperature_sweep')
    old_to_new_id = launchpad.add_wf(workflow)

    fw_ids = {
        temperature_by_placeholder_id[old_id]: new_id
        for old_id, new_id in old_to_new_id.items()
        if old_id in temperature_by_placeholder_id
    }

    return {
        'fw_ids': fw_ids,
        'wf_id': next(iter(fw_ids.values()), None),
        'launchpad': str(launchpad),
    }


def check_fireworks_status(fw_ids: list) -> dict:
    """
    Query a FireWorks LaunchPad for the current state of one or more previously
    submitted Fireworks, such as the ones returned by
    submit_temperature_sweep_to_fireworks.

    Parameters
    ----------
    fw_ids : list[int] -- Firework ids to check, as returned in the "fw_ids"
                           dict of submit_temperature_sweep_to_fireworks

    Returns
    -------
    dict -- {
        fw_id: {
            "state": one of FireWorks' states (READY, RUNNING, COMPLETED,
                     FIZZLED, etc.),
            "launch_dir": str or None -- working directory of the most recent
                          launch, if the Firework has started,
        },
        ...
    }
    """
    from fireworks import LaunchPad

    launchpad = LaunchPad.auto_load()

    status = {}
    for fw_id in fw_ids:
        fw = launchpad.get_fw_by_id(fw_id)
        launches = fw.launches + fw.archived_launches
        launch_dir = launches[-1].launch_dir if launches else None
        status[fw_id] = {
            'state': fw.state,
            'launch_dir': launch_dir,
        }
    return status
