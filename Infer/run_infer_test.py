import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


DEFAULT_SPE = Path(r"D:\dataset\2025-11月\1122\1122\6022025112200000202.spe")
DEFAULT_HDR = Path(r"D:\dataset\2025-11月\1122\1122\6022025112200000202.hdr")
DEFAULT_MODEL_DIR = Path(r"..\models")
DEFAULT_7X5_CENTERS = [
    [(282, 54), (218, 54), (153, 54), (92, 54), (26, 54)],
    [(282, 114), (219, 114), (156, 114), (91, 114), (27, 114)],
    [(282, 179), (219, 179), (153, 179), (92, 179), (27, 179)],
    [(282, 239), (219, 239), (153, 239), (91, 239), (27, 239)],
    [(282, 301), (219, 301), (153, 301), (91, 301), (27, 301)],
    [(282, 363), (219, 363), (153, 363), (91, 363), (27, 363)],
    [(282, 426), (219, 426), (153, 426), (91, 426), (27, 426)],
]
ROW_GROUP_TOLERANCE = 10


def _safe_display(value):
    return str(value).encode("unicode_escape").decode("ascii")


def _linspace_int(start, stop, count):
    if count <= 0:
        return []
    if count == 1:
        return [int(round(start))]

    step = (stop - start) / (count - 1)
    return [int(round(start + index * step)) for index in range(count)]


def _build_grid_centers(rows, cols, x_right=282, x_left=24, y_top=54, y_bottom=426):
    x_values = _linspace_int(x_right, x_left, cols)
    y_values = _linspace_int(y_top, y_bottom, rows)
    return [[(x, y) for x in x_values] for y in y_values]


def _parse_center_position(center_position):
    if center_position is None:
        return None
    if isinstance(center_position, (list, tuple)):
        return center_position
    if isinstance(center_position, str):
        stripped = center_position.strip()
        if stripped.startswith("center_position"):
            _, _, stripped = stripped.partition("=")
            stripped = stripped.strip()
        if stripped.startswith(("[", "(", "{")):
            try:
                return ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return json.loads(stripped)

        center_path = Path(center_position)
        if center_path.exists():
            return json.loads(center_path.read_text(encoding="utf-8"))
    raise TypeError("Unsupported center_position payload")


def _flatten_centers(center_position):
    centers = _parse_center_position(center_position)
    if centers is None:
        centers = DEFAULT_7X5_CENTERS

    flattened = []
    for item in centers:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
            flattened.extend((int(x), int(y)) for x, y in item)
        else:
            x, y = item
            flattened.append((int(x), int(y)))
    return flattened


def _sort_centers(center_position):
    return sorted(_flatten_centers(center_position), key=lambda item: (item[1], -item[0]))


def _infer_grid_shape(center_position):
    centers = _parse_center_position(center_position)
    if centers is None:
        return len(DEFAULT_7X5_CENTERS), len(DEFAULT_7X5_CENTERS[0])

    if (
        isinstance(centers, (list, tuple))
        and centers
        and isinstance(centers[0], (list, tuple))
        and centers[0]
        and isinstance(centers[0][0], (list, tuple))
    ):
        row_lengths = [len(row) for row in centers]
        if any(length != row_lengths[0] for length in row_lengths):
            raise ValueError(f"center_position is not rectangular: {row_lengths}")
        return len(row_lengths), row_lengths[0]

    sorted_centers = _sort_centers(centers)
    row_groups = [[sorted_centers[0]]]
    for point in sorted_centers[1:]:
        group = row_groups[-1]
        mean_y = sum(y for _, y in group) / len(group)
        if abs(point[1] - mean_y) <= ROW_GROUP_TOLERANCE:
            group.append(point)
        else:
            row_groups.append([point])

    row_lengths = [len(group) for group in row_groups]
    if any(length != row_lengths[0] for length in row_lengths):
        raise ValueError(f"center_position is not rectangular: {row_lengths}")
    return len(row_groups), row_lengths[0]


def _load_infer_module(script_dir):
    exact_pyd = script_dir / "EggInfer.pyd"
    exact_py = script_dir / "EggInfer.py"
    if exact_pyd.exists():
        matches = [exact_pyd]
    elif exact_py.exists():
        matches = [exact_py]
    else:
        matches = sorted(script_dir.glob("EggInfer*.pyd"))
        if not matches:
            matches = sorted(script_dir.glob("EggInfer*.py"))
    if not matches:
        raise FileNotFoundError(f"No EggInfer*.py or EggInfer*.pyd found under {script_dir}")

    entry_path = matches[0]
    module_name = entry_path.stem
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(module_name, entry_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, entry_path


def _build_parser():
    parser = argparse.ArgumentParser(description="Run a local inference smoke test.")
    parser.add_argument("--spe", type=Path, default=DEFAULT_SPE, help="Path to .spe file")
    parser.add_argument("--hdr", type=Path, default=DEFAULT_HDR, help="Path to .hdr file")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing one model set (*.pt files)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Single threshold passed to identifyEggs",
    )
    parser.add_argument(
        "--mem-key",
        default=None,
        help="Pallet key written into the output JSON, defaults to SPE stem",
    )
    parser.add_argument(
        "--center-position",
        default=None,
        help=(
            "Optional center_position payload. Supports Python/JSON text, "
            "or a JSON file path handled by EggInfer.identifyEggs."
        ),
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Optional passphrase for loading encrypted *.pt.enc model files.",
    )
    parser.add_argument(
        "--case",
        choices=["all", "7x6", "6x7", "5x7", "7x5"],
        default="all",
        help="Built-in center_position case(s) to run when --center-position is not provided.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the final JSON result",
    )
    return parser


def _build_builtin_cases():
    return [
        ("7x6", _build_grid_centers(7, 6)),
        ("6x7", _build_grid_centers(6, 7)),
        ("5x7", _build_grid_centers(5, 7)),
        ("7x5", DEFAULT_7X5_CENTERS),
    ]


def _build_out_path(out_path, case_name, multi_case):
    if out_path is None:
        return None

    resolved = out_path.resolve()
    if not multi_case:
        return resolved
    if resolved.suffix:
        return resolved.with_name(f"{resolved.stem}_{case_name}{resolved.suffix}")
    return resolved.with_name(f"{resolved.name}_{case_name}.json")


def _normalize_center_rows(center_position):
    rows, cols = _infer_grid_shape(center_position)
    sorted_centers = _sort_centers(center_position)
    row_groups = []
    for row_index in range(rows):
        start = row_index * cols
        end = start + cols
        row_groups.append([[x, y] for x, y in sorted_centers[start:end]])
    return row_groups


def _format_center_rows(center_position):
    row_groups = _normalize_center_rows(center_position)
    lines = ["["]
    for row_index, row in enumerate(row_groups):
        row_text = ", ".join(f"[{x}, {y}]" for x, y in row)
        suffix = "," if row_index < len(row_groups) - 1 else ""
        lines.append(f"  [{row_text}]{suffix}")
    lines.append("]")
    return "\n".join(lines)


def _run_case(
    infer_module,
    case_name,
    center_position,
    spe_bytes,
    hdr_bytes,
    model_dir,
    mem_key,
    threshold,
    model_key=None,
    out_path=None,
):
    grid_shape = _infer_grid_shape(center_position)
    case_mem_key = mem_key if case_name == "custom" else f"{mem_key}_{case_name}"

    print(f"[TEST] case={case_name}")
    print(f"[TEST] grid_shape={grid_shape}")
    print(
        f"[TEST] center_position={'custom' if case_name == 'custom' else 'builtin'}"
    )
    print("[TEST] center_values:")
    print(_format_center_rows(center_position))

    result_json = infer_module.identifyEggs(
        spe_bytes,
        hdr_bytes,
        str(model_dir),
        mem_key=case_mem_key,
        thresholds=[threshold],
        center_position=center_position,
        model_key=model_key,
    )

    parsed = json.loads(result_json)
    print("[TEST] result_json:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if out_path is not None:
        out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[TEST] wrote_result={_safe_display(out_path)}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    infer_module, entry_path = _load_infer_module(script_dir)

    spe_path = args.spe.resolve()
    hdr_path = args.hdr.resolve()
    model_dir = args.model_dir.resolve()
    mem_key = args.mem_key or spe_path.stem

    if not spe_path.exists():
        raise FileNotFoundError(f"SPE file not found: {spe_path}")
    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"[TEST] entry={_safe_display(entry_path)}")
    print(f"[TEST] spe={_safe_display(spe_path)}")
    print(f"[TEST] hdr={_safe_display(hdr_path)}")
    print(f"[TEST] model_dir={_safe_display(model_dir)}")
    print(f"[TEST] threshold={args.threshold}")
    print(f"[TEST] mem_key={mem_key}")
    print(f"[TEST] model_key={'provided' if args.model_key else 'builtin/default'}")
    print(f"[TEST] case_mode={'custom' if args.center_position else args.case}")

    spe_bytes = spe_path.read_bytes()
    hdr_bytes = hdr_path.read_bytes()

    if args.center_position is not None:
        cases = [("custom", args.center_position)]
    else:
        builtin_cases = _build_builtin_cases()
        if args.case == "all":
            cases = builtin_cases
        else:
            cases = [item for item in builtin_cases if item[0] == args.case]

    multi_case = len(cases) > 1
    for case_name, center_position in cases:
        case_out = _build_out_path(args.out, case_name, multi_case)
        _run_case(
            infer_module=infer_module,
            case_name=case_name,
            center_position=center_position,
            spe_bytes=spe_bytes,
            hdr_bytes=hdr_bytes,
            model_dir=model_dir,
            mem_key=mem_key,
            threshold=args.threshold,
            model_key=args.model_key,
            out_path=case_out,
        )


if __name__ == "__main__":
    main()
