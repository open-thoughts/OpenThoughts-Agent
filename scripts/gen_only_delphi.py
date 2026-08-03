"""Generate the canonical SFT sbatch WITHOUT submitting.
Monkeypatches launch_sbatch to capture the sbatch path instead of calling sbatch.
Usage: python gen_only_delphi.py <argstring...>  (same flags as hpc.launch, minus --dry_run)
The captured sbatch path is printed as: GENERATED_SBATCH=<path>
"""
import hpc.launch_utils as lu
import hpc.sft_launch_utils as slu
import hpc.launch as launch

_captured = {}
def fake_launch_sbatch(sbatch_script_path, dependency=None, array=None, **kw):
    _captured["path"] = str(sbatch_script_path)
    _captured["dependency"] = dependency
    print(f"GENERATED_SBATCH={sbatch_script_path}")
    print(f"NOSUBMIT_DEPENDENCY={dependency}")
    return "NOSUBMIT"

# Patch every reference
lu.launch_sbatch = fake_launch_sbatch
launch.launch_sbatch = fake_launch_sbatch
if hasattr(slu, "launch_sbatch"):
    slu.launch_sbatch = fake_launch_sbatch

# hpc.launch.main reads sys.argv
launch.main()
