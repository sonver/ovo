import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True


DEFAULT_RELEASE_PREFIX = "release_pyarmor_singlefile_"
DEFAULT_SOURCE_MODEL_DIR = Path("model") / "0413"
DEFAULT_SPE = Path("D:/dataset/2025-11\u6708/1122/1122/6022025112200000203.spe")
DEFAULT_HDR = Path("D:/dataset/2025-11\u6708/1122/1122/6022025112200000203.hdr")
DEFAULT_CASE = "7x5"
DEFAULT_THRESHOLD = "0.4"
PACKAGE_DIR_NAME = "package"


def _safe_display(value):
    return str(value).encode("unicode_escape").decode("ascii")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Verify that a published Infer release matches the source inference output."
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="Release directory to verify. Defaults to the latest release_pyarmor_singlefile_* directory.",
    )
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=DEFAULT_SOURCE_MODEL_DIR,
        help="Plain source model directory used for source-mode verification.",
    )
    parser.add_argument("--spe", type=Path, default=DEFAULT_SPE, help="Path to .spe file")
    parser.add_argument("--hdr", type=Path, default=DEFAULT_HDR, help="Path to .hdr file")
    parser.add_argument(
        "--case",
        choices=["7x6", "6x7", "5x7", "7x5"],
        default=DEFAULT_CASE,
        help="Built-in center-position case passed to both source and release tests.",
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD,
        help="Threshold passed to both source and release tests.",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Optional release model key override. Normally not needed for the bundled release models.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the temporary JSON artifacts used for comparison.",
    )
    return parser


def _find_latest_release(script_dir):
    candidates = sorted(
        path
        for path in script_dir.iterdir()
        if path.is_dir() and path.name.startswith(DEFAULT_RELEASE_PREFIX)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No release directory found under {script_dir} with prefix {DEFAULT_RELEASE_PREFIX}"
        )
    return candidates[-1]


def _require_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _resolve_package_dir(release_dir):
    package_dir = release_dir / PACKAGE_DIR_NAME
    if package_dir.exists():
        return package_dir
    return release_dir


def _check_release_structure(release_dir):
    package_dir = _resolve_package_dir(release_dir)
    required_paths = [
        (package_dir / "code" / "EggInfer.py", "Protected EggInfer entry"),
        (package_dir / "code" / "run_infer_test.py", "Release smoke test script"),
        (package_dir / "code" / "pyarmor_runtime_000000", "PyArmor runtime directory"),
        (package_dir / "models", "Encrypted model directory"),
        (package_dir / "README.txt", "Release README"),
    ]
    for path, description in required_paths:
        _require_path(path, description)

    if package_dir != release_dir:
        _require_path(release_dir / "README_INTERNAL.txt", "Internal release README")
        _require_path(release_dir / f"{release_dir.name}.zip", "Release archive")

    encrypted_models = sorted((package_dir / "models").glob("*.pt.enc"))
    if not encrypted_models:
        raise FileNotFoundError(
            f"No encrypted *.pt.enc models found under {package_dir / 'models'}"
        )
    return package_dir, encrypted_models


def _run(command, workdir):
    print(f"[VERIFY] {' '.join(_safe_display(part) for part in command)}")
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(map(str, command))}"
        )
    return completed.stdout


def _extract_result_json(stdout):
    marker = "[TEST] result_json:"
    if marker not in stdout:
        raise ValueError("Could not find '[TEST] result_json:' in command output")
    payload = stdout.split(marker, 1)[1].strip()
    return json.loads(payload)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python_exe = Path(sys.executable)
    release_dir = args.release_dir.resolve() if args.release_dir else _find_latest_release(script_dir)
    source_model_dir = (script_dir / args.source_model_dir).resolve()
    spe_path = args.spe.resolve()
    hdr_path = args.hdr.resolve()

    _require_path(release_dir, "Release directory")
    _require_path(source_model_dir, "Source model directory")
    _require_path(spe_path, "SPE file")
    _require_path(hdr_path, "HDR file")
    package_dir, encrypted_models = _check_release_structure(release_dir)

    print(f"[VERIFY] release_dir={_safe_display(release_dir)}")
    print(f"[VERIFY] package_dir={_safe_display(package_dir)}")
    print(f"[VERIFY] source_model_dir={_safe_display(source_model_dir)}")
    print(f"[VERIFY] encrypted_model_count={len(encrypted_models)}")

    source_command = [
        str(python_exe),
        str(script_dir / "run_infer_test.py"),
        "--case",
        args.case,
        "--spe",
        str(spe_path),
        "--hdr",
        str(hdr_path),
        "--model-dir",
        str(source_model_dir),
        "--threshold",
        str(args.threshold),
    ]
    release_command = [
        str(python_exe),
        str(package_dir / "code" / "run_infer_test.py"),
        "--case",
        args.case,
        "--spe",
        str(spe_path),
        "--hdr",
        str(hdr_path),
        "--model-dir",
        str(package_dir / "models"),
        "--threshold",
        str(args.threshold),
    ]
    if args.model_key:
        release_command.extend(["--model-key", args.model_key])

    source_stdout = _run(source_command, script_dir)
    release_stdout = _run(release_command, script_dir)

    source_payload = _extract_result_json(source_stdout)
    release_payload = _extract_result_json(release_stdout)
    same_json = source_payload == release_payload

    source_stats = source_payload.get("statistics", {})
    release_stats = release_payload.get("statistics", {})
    print(f"[VERIFY] source_statistics={source_stats}")
    print(f"[VERIFY] release_statistics={release_stats}")
    print(f"[VERIFY] same_json={same_json}")

    if not same_json:
        raise AssertionError("Release output JSON does not match source output JSON")

    if args.keep_artifacts:
        artifact_root = release_dir / "_verify_artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "source.json").write_text(
            json.dumps(source_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifact_root / "release.json").write_text(
            json.dumps(release_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("[VERIFY] release verification passed")


if __name__ == "__main__":
    main()
