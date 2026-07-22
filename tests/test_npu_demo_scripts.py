import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo_scripts"
NPU_DEMO_DIR = ROOT / "npu_demo_scripts"


def test_npu_demo_scripts_cover_all_demos():
    demos = {path.name for path in DEMO_DIR.glob("*.sh")}
    npu_demos = {path.name for path in NPU_DEMO_DIR.glob("*.sh")}

    assert npu_demos == demos


def test_npu_demo_scripts_use_npu_defaults_and_valid_shell():
    for script in NPU_DEMO_DIR.glob("*.sh"):
        content = script.read_text(encoding="utf-8")

        assert 'export device="npu:' in content
        assert "cuda" not in content
        assert "python inference.py" in content
        subprocess.run(["bash", "-n", script], check=True)
