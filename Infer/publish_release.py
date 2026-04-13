import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True


FIXED_MODEL_KEY = "WL-20260226"
DEFAULT_TOOL = "pyarmor"
DEFAULT_RELEASE_PREFIX = "release_pyarmor_singlefile"
DEFAULT_MODEL_DIR = Path("model") / "0413"
DEFAULT_SPE = Path("D:/dataset/2025-11\u6708/1122/1122/6022025112200000202.spe")
DEFAULT_HDR = Path("D:/dataset/2025-11\u6708/1122/1122/6022025112200000202.hdr")
DEFAULT_CASE = "7x5"
PACKAGE_DIR_NAME = "package"


def _safe_display(value):
    return str(value).encode("unicode_escape").decode("ascii")


def _run(command, workdir):
    print(f"[PUBLISH] {' '.join(_safe_display(part) for part in command)}")
    subprocess.run(command, check=True, cwd=str(workdir))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Auto-publish the Infer release package with a fixed model key."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Plain model directory to encrypt into the release. Default: Infer\\model\\0413",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_RELEASE_PREFIX,
        help=f"Release directory prefix. Default: {DEFAULT_RELEASE_PREFIX}",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the post-build smoke test.",
    )
    parser.add_argument(
        "--spe",
        type=Path,
        default=DEFAULT_SPE,
        help="Smoke-test .spe path. Used unless --skip-test is set.",
    )
    parser.add_argument(
        "--hdr",
        type=Path,
        default=DEFAULT_HDR,
        help="Smoke-test .hdr path. Used unless --skip-test is set.",
    )
    return parser


def _iter_source_model_names(model_dir):
    if not model_dir.exists():
        return []
    return sorted(path.name for path in model_dir.iterdir() if path.is_file())


def _prepare_package_layout(release_dir):
    package_dir = release_dir / PACKAGE_DIR_NAME
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in ("code", "models"):
        source = release_dir / name
        target = package_dir / name
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(source), str(target))
    return package_dir


def _build_archive(package_dir, archive_path):
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(package_dir.parent)))


def _write_public_readme(readme_path, package_dir):
    package_dir_str = str(package_dir)
    content = "\n".join(
        [
            "Infer release package",
            "",
            "Directory layout",
            "- code\\EggInfer.py : protected inference entry",
            "- code\\run_infer_test.py : plaintext smoke test script",
            "- models\\*.pt.enc : encrypted model files",
            "",
            "Environment",
            "- Python: C:\\Users\\liu\\Anaconda3\\envs\\py312\\python.exe",
            "- Required packages: torch, opencv-python/cv2, scipy, osgeo/gdal",
            "",
            "Smoke test",
            f'& "C:\\Users\\liu\\Anaconda3\\envs\\py312\\python.exe" "{package_dir_str}\\code\\run_infer_test.py" --case {DEFAULT_CASE} --model-dir "{package_dir_str}\\models"',
            "",
        ]
    )
    readme_path.write_text(content, encoding="utf-8")


def _write_internal_readme(
    readme_path, release_dir, package_dir, archive_path, model_dir, python_exe, timestamp
):
    release_dir_str = str(release_dir)
    package_dir_str = str(package_dir)
    archive_path_str = str(archive_path)
    source_models = _iter_source_model_names(model_dir)
    model_lines = ["Bundled source model files"]
    if source_models:
        model_lines.extend(f"- {name}" for name in source_models)
    else:
        model_lines.append("- <none found>")

    content = "\n".join(
        [
            "Infer internal release record",
            "",
            f"Release directory: {release_dir_str}",
            f"Deliverable directory: {package_dir_str}",
            f"Archive path: {archive_path_str}",
            f"Publish time: {timestamp}",
            f"Publish script: {Path(__file__).name}",
            f"Build tool: {DEFAULT_TOOL}",
            f"Python: {python_exe}",
            f"Model source: {model_dir}",
            f"Built-in model key: {FIXED_MODEL_KEY}",
            f"Smoke-test SPE: {DEFAULT_SPE}",
            f"Smoke-test HDR: {DEFAULT_HDR}",
            f"Smoke-test case: {DEFAULT_CASE}",
            "",
            "Customer-facing README",
            f"- {package_dir_str}\\README.txt",
            "",
            *model_lines,
            "",
            "Re-encrypt new models into this release directory",
            f'& "{python_exe}" "D:\\gpt-codex\\ovo\\Infer\\EggInfer.py" encrypt-models --input-dir "{model_dir}" --output-dir "{package_dir_str}\\models" --key "{FIXED_MODEL_KEY}" --overwrite',
            "",
            "Release smoke test",
            f'& "{python_exe}" "{package_dir_str}\\code\\run_infer_test.py" --case {DEFAULT_CASE} --model-dir "{package_dir_str}\\models"',
            "",
            "Notes",
            "- run_infer_test.py is intentionally kept plaintext.",
            "- Bundled release models use the built-in key above, so normal testing does not need --model-key.",
            "- If models are re-encrypted with a different key, pass --model-key or set EGG_INFER_MODEL_KEY.",
            "",
        ]
    )
    readme_path.write_text(content, encoding="utf-8")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python_exe = Path(sys.executable)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_dir = script_dir / f"{args.prefix}_{timestamp}"
    model_dir = (script_dir / args.model_dir).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    build_command = [
        str(python_exe),
        str(script_dir / "build_protected_infer.py"),
        "--output-dir",
        str(release_dir),
        "--tool",
        DEFAULT_TOOL,
        "--include-test",
        "--encrypt-model-source",
        str(model_dir),
        "--model-key",
        FIXED_MODEL_KEY,
    ]
    _run(build_command, script_dir)

    package_dir = _prepare_package_layout(release_dir)
    archive_path = release_dir / f"{release_dir.name}.zip"

    _write_public_readme(package_dir / "README.txt", package_dir)
    _write_internal_readme(
        release_dir / "README_INTERNAL.txt",
        release_dir,
        package_dir,
        archive_path,
        model_dir,
        python_exe,
        timestamp,
    )
    print(f"[PUBLISH] wrote README: {_safe_display(package_dir / 'README.txt')}")
    print(
        f"[PUBLISH] wrote README: {_safe_display(release_dir / 'README_INTERNAL.txt')}"
    )

    if not args.skip_test:
        spe_path = args.spe.resolve()
        hdr_path = args.hdr.resolve()
        if not spe_path.exists():
            raise FileNotFoundError(f"Smoke-test SPE not found: {spe_path}")
        if not hdr_path.exists():
            raise FileNotFoundError(f"Smoke-test HDR not found: {hdr_path}")

        test_command = [
            str(python_exe),
            str(package_dir / "code" / "run_infer_test.py"),
            "--case",
            DEFAULT_CASE,
            "--spe",
            str(spe_path),
            "--hdr",
            str(hdr_path),
            "--model-dir",
            str(package_dir / "models"),
        ]
        _run(test_command, script_dir)

    _build_archive(package_dir, archive_path)
    print(f"[PUBLISH] archive={_safe_display(archive_path)}")
    print(f"[PUBLISH] release_dir={_safe_display(release_dir)}")
    print(f"[PUBLISH] package_dir={_safe_display(package_dir)}")
    print(f"[PUBLISH] model_key={FIXED_MODEL_KEY}")


if __name__ == "__main__":
    main()
