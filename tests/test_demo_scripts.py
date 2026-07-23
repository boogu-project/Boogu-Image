import os
import re
import subprocess
import sys
from pathlib import Path

from inference import parse_args


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo_scripts"


def test_demo_scripts_use_overrideable_devices_and_valid_shell():
    scripts = sorted(DEMO_DIR.glob("*.sh"))
    assert scripts

    for script in scripts:
        content = script.read_text(encoding="utf-8")
        assert re.search(
            r'^export device="\$\{device:-cuda:\d+\}"$', content, re.MULTILINE
        )
        assert re.search(
            r'^export rewriter_device="\$\{rewriter_device:-(?:\$device|cuda:\d+)\}"',
            content,
            re.MULTILINE,
        )
        assert "python inference.py" in content
        subprocess.run(["bash", "-n", script], check=True)


def test_demo_scripts_expand_npu_arguments(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python"
    python_stub.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGV_LOG"\n', encoding="utf-8"
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "ARGV_LOG": str(tmp_path / "argv.log"),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "device": "npu:0",
            "rewriter_device": "npu:1",
        }
    )

    for script in sorted(DEMO_DIR.glob("*.sh")):
        run_dir = tmp_path / script.stem
        run_dir.mkdir()
        subprocess.run(["bash", script], check=True, cwd=run_dir, env=env)

        argv = (tmp_path / "argv.log").read_text(encoding="utf-8").splitlines()
        monkeypatch.setattr(sys, "argv", argv)
        args = parse_args()
        assert args.device == "npu:0"
        assert args.rewriter_device == "npu:1"
