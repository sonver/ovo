import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _find_entry(script_dir):
    exact = script_dir / "EggInfer.py"
    if exact.exists():
        return exact

    matches = sorted(script_dir.glob("EggInfer*.py"))
    if not matches:
        raise FileNotFoundError(f"No EggInfer*.py found under {script_dir}")
    return matches[0]


def _module_exists(name):
    return importlib.util.find_spec(name) is not None


def _find_executable(candidates):
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _resolve_pyarmor_executable(python_exe):
    candidate = _find_executable(["pyarmor"])
    if candidate:
        return candidate

    for name in ("pyarmor.exe", "pyarmor.cmd", "pyarmor.bat"):
        path = python_exe.parent / "Scripts" / name
        if path.exists():
            return str(path)
    return None


def _resolve_nuitka_invocation(python_exe):
    candidate = _find_executable(["nuitka"])
    if candidate:
        return [candidate]

    for name in ("nuitka.exe", "nuitka.cmd", "nuitka.bat"):
        path = python_exe.parent / "Scripts" / name
        if path.exists():
            return [str(path)]

    if _module_exists("nuitka"):
        return [str(python_exe), "-m", "nuitka"]
    return None


def _tool_available(tool_name, python_exe):
    if tool_name == "pyarmor":
        return _resolve_pyarmor_executable(python_exe) is not None or _module_exists("pyarmor")
    if tool_name == "nuitka":
        return _resolve_nuitka_invocation(python_exe) is not None
    return False


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build an obfuscated/protected Infer package."
    )
    parser.add_argument(
        "--tool",
        choices=["pyarmor", "nuitka"],
        default="pyarmor",
        help="Protection tool used for the Python code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Build output directory.",
    )
    parser.add_argument(
        "--copy-model-dir",
        type=Path,
        default=None,
        help="Optional model directory copied into the build output as-is.",
    )
    parser.add_argument(
        "--encrypt-model-source",
        type=Path,
        default=None,
        help="Optional plain model directory to encrypt into the build output.",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Passphrase used when --encrypt-model-source is provided.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Copy run_infer_test.py into the build output without obfuscating it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build commands without executing them.",
    )
    return parser


def _quoted(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def _safe_display(value):
    return str(value).encode("unicode_escape").decode("ascii")


def _run(command, dry_run, env_updates=None):
    print(f"[BUILD] {_safe_display(_quoted(command))}")
    if not dry_run:
        env = os.environ.copy()
        if env_updates:
            env.update(env_updates)
        subprocess.run(command, check=True, env=env)


def _copy_tree(source_dir, destination_dir, dry_run):
    print(f"[BUILD] copy {_safe_display(source_dir)} -> {_safe_display(destination_dir)}")
    if dry_run:
        return
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def _build_pyarmor_command(python_exe, output_dir, file_paths):
    pyarmor_exe = _resolve_pyarmor_executable(python_exe)
    if pyarmor_exe is None:
        return [
            str(python_exe),
            "-m",
            "pyarmor",
            "gen",
            "--recursive",
            "-O",
            str(output_dir),
            *[str(path) for path in file_paths],
        ]
    pyarmor_home = output_dir.parent / ".pyarmor_home"
    return [
        pyarmor_exe,
        "--home",
        str(pyarmor_home),
        "gen",
        "--recursive",
        "-O",
        str(output_dir),
        *[str(path) for path in file_paths],
    ]


def _build_nuitka_command(python_exe, output_dir, source_path):
    nuitka_command = _resolve_nuitka_invocation(python_exe)
    if nuitka_command is None:
        nuitka_command = [str(python_exe), "-m", "nuitka"]
    return [
        *nuitka_command,
        "--module",
        "--assume-yes-for-downloads",
        f"--output-dir={output_dir}",
        str(source_path),
    ]


def _copy_file(source_path, destination_dir, dry_run):
    destination_dir = Path(destination_dir)
    print(f"[BUILD] copy {_safe_display(source_path)} -> {_safe_display(destination_dir / source_path.name)}")
    if dry_run:
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_dir / source_path.name)


def _is_self_contained_entry(entry_path):
    return entry_path.name == "EggInfer.py"


def _prune_release_artifacts(output_dir, dry_run):
    prune_targets = [output_dir / ".pyarmor_home"]
    prune_targets.extend(output_dir.rglob("__pycache__"))

    for target in prune_targets:
        if not target.exists():
            continue
        print(f"[BUILD] prune {_safe_display(target)}")
        if not dry_run:
            shutil.rmtree(target, ignore_errors=True)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_path = _find_entry(script_dir)
    python_exe = Path(sys.executable)
    file_paths = [entry_path]
    if not _is_self_contained_entry(entry_path):
        file_paths.extend(
            [
                script_dir / "tools.py",
                script_dir / "model_crypto.py",
                script_dir / "encrypt_models.py",
            ]
        )

    if not _tool_available(args.tool, python_exe) and not args.dry_run:
        raise ModuleNotFoundError(
            f"{args.tool} is not available in {python_exe}. "
            f"Install it first, then rerun this build script."
        )
    if not _tool_available(args.tool, python_exe) and args.dry_run:
        print(
            f"[BUILD] warning: {args.tool} is not available in {python_exe}. "
            f"Dry-run will print commands only."
        )

    code_output_dir = output_dir / "code"
    if args.tool == "pyarmor":
        command = _build_pyarmor_command(python_exe, code_output_dir, file_paths)
        _run(command, args.dry_run)
        if args.include_test:
            _copy_file(script_dir / "run_infer_test.py", code_output_dir, args.dry_run)
    else:
        code_output_dir.mkdir(parents=True, exist_ok=True)
        nuitka_cache_dir = output_dir / ".nuitka_cache"
        nuitka_targets = [entry_path]
        if not _is_self_contained_entry(entry_path):
            nuitka_targets.extend(
                [
                    script_dir / "tools.py",
                    script_dir / "model_crypto.py",
                ]
            )
        for target in nuitka_targets:
            _run(
                _build_nuitka_command(python_exe, code_output_dir, target),
                args.dry_run,
                env_updates={"NUITKA_CACHE_DIR": str(nuitka_cache_dir)},
            )
            if not args.dry_run and not any(code_output_dir.glob(f"{target.stem}*.pyd")):
                raise RuntimeError(
                    f"Nuitka build did not produce a .pyd for {target.name} under {code_output_dir}"
                )

        if not _is_self_contained_entry(entry_path):
            _copy_file(script_dir / "encrypt_models.py", code_output_dir, args.dry_run)
        if args.include_test:
            _copy_file(script_dir / "run_infer_test.py", code_output_dir, args.dry_run)

    if args.copy_model_dir is not None:
        model_source = args.copy_model_dir.resolve()
        if not model_source.exists():
            raise FileNotFoundError(f"copy-model-dir not found: {model_source}")
        _copy_tree(model_source, output_dir / "models", args.dry_run)

    if args.encrypt_model_source is not None:
        plain_model_dir = args.encrypt_model_source.resolve()
        if not plain_model_dir.exists():
            raise FileNotFoundError(
                f"encrypt-model-source not found: {plain_model_dir}"
            )
        if not args.model_key:
            raise ValueError("--model-key is required with --encrypt-model-source")

        if _is_self_contained_entry(entry_path):
            encrypt_command = [
                str(python_exe),
                str(entry_path),
                "encrypt-models",
                "--input-dir",
                str(plain_model_dir),
                "--output-dir",
                str(output_dir / "models"),
                "--key",
                args.model_key,
                "--recursive",
                "--overwrite",
            ]
        else:
            encrypt_command = [
                str(python_exe),
                str(script_dir / "encrypt_models.py"),
                "--input-dir",
                str(plain_model_dir),
                "--output-dir",
                str(output_dir / "models"),
                "--key",
                args.model_key,
                "--recursive",
                "--overwrite",
            ]
        _run(encrypt_command, args.dry_run)

    _prune_release_artifacts(output_dir, args.dry_run)


if __name__ == "__main__":
    main()
