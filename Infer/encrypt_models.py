"""Compatibility CLI shim for the consolidated single-file EggInfer module."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from EggInfer import DEFAULT_KEY_ENV, TARGET_PATTERNS, encrypt_model_directory

__all__ = ["DEFAULT_KEY_ENV", "TARGET_PATTERNS", "encrypt_model_directory", "main"]


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Encrypt TorchScript model files and companion temperature files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .pt models and optional *_temperature.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for encrypted *.enc files.",
    )
    parser.add_argument("--key", default=None, help="Passphrase used for encryption.")
    parser.add_argument(
        "--key-env",
        default=DEFAULT_KEY_ENV,
        help=f"Environment variable used when --key is omitted. Default: {DEFAULT_KEY_ENV}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Encrypt matching files under nested model-set directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing encrypted output files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults to <output-dir>/encryption_manifest.json.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    encrypt_model_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        key=args.key,
        key_env=args.key_env,
        recursive=args.recursive,
        overwrite=args.overwrite,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
