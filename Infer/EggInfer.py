import argparse
import ast
import hashlib
import hmac
import io
import json
import os
import secrets
import struct
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ModuleNotFoundError:
    AESGCM = None

try:
    from osgeo import gdal
except ModuleNotFoundError:
    gdal = None


IMAGES_CENTER = [
    [(282, 425), (218, 425), (153, 425), (92, 425), (26, 425)],
    [(282, 365), (219, 365), (156, 365), (91, 365), (27, 365)],
    [(282, 300), (219, 300), (153, 300), (92, 300), (27, 300)],
    [(282, 240), (219, 240), (153, 240), (91, 240), (27, 240)],
    [(282, 178), (219, 178), (153, 178), (91, 178), (27, 178)],
    [(282, 116), (219, 116), (153, 116), (91, 116), (27, 116)],
    [(282, 53), (219, 53), (153, 53), (91, 53), (27, 53)],
]
ADJUSTED_IMAGES_CENTER = [[(x, 479 - y) for x, y in row] for row in IMAGES_CENTER]
ROW_GROUP_TOLERANCE = 10
MODEL_KEY_ENV = "EGG_INFER_MODEL_KEY"
BUILTIN_MODEL_KEY = "WL-20260226"
DEFAULT_KEY_ENV = MODEL_KEY_ENV
TARGET_PATTERNS = ("*.pt", "*_temperature.txt")

MAGIC = b"EGGMOD2\0"
VERSION = 2
PBKDF2_ITERATIONS = 300000
SALT_LEN = 16
NONCE_LEN = 12
HEADER_FORMAT = ">8sBI16s12sQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

LEGACY_MAGIC = b"EGGMOD1\0"
LEGACY_VERSION = 1
LEGACY_NONCE_LEN = 16
LEGACY_TAG_LEN = 32
LEGACY_HEADER_FORMAT = ">8sBI16s16sQ"
LEGACY_HEADER_SIZE = struct.calcsize(LEGACY_HEADER_FORMAT)


class _FallbackBand:
    def __init__(self, dataset, band_index):
        self._dataset = dataset
        self._band_index = band_index

    def ReadAsArray(self):
        return np.asarray(self._dataset._data[:, self._band_index, :])


class _FallbackEnviDataset:
    def __init__(self, spe_file, header):
        self._spe_file = spe_file
        self._header = header
        self.RasterCount = int(header["bands"])
        self._data = self._open_memmap()

    def _dtype_from_header(self):
        data_type = int(self._header["data type"])
        byte_order = int(self._header.get("byte order", 0))
        mapping = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            12: np.uint16,
        }
        if data_type not in mapping:
            raise ValueError(f"Unsupported ENVI data type: {data_type}")

        dtype = np.dtype(mapping[data_type])
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder("<" if byte_order == 0 else ">")
        return dtype

    def _open_memmap(self):
        samples = int(self._header["samples"])
        lines = int(self._header["lines"])
        bands = int(self._header["bands"])
        interleave = self._header.get("interleave", "").strip().lower()
        header_offset = int(self._header.get("header offset", 0))
        dtype = self._dtype_from_header()

        if interleave == "bil":
            return np.memmap(
                self._spe_file,
                dtype=dtype,
                mode="r",
                offset=header_offset,
                shape=(lines, bands, samples),
            )
        if interleave == "bip":
            data = np.memmap(
                self._spe_file,
                dtype=dtype,
                mode="r",
                offset=header_offset,
                shape=(lines, samples, bands),
            )
            return np.transpose(data, (0, 2, 1))
        if interleave == "bsq":
            data = np.memmap(
                self._spe_file,
                dtype=dtype,
                mode="r",
                offset=header_offset,
                shape=(bands, lines, samples),
            )
            return np.transpose(data, (1, 0, 2))
        raise ValueError(f"Unsupported ENVI interleave: {interleave}")

    def GetRasterBand(self, band_index):
        return _FallbackBand(self, band_index - 1)


class _GdalShim:
    @staticmethod
    def UseExceptions():
        return None

    @staticmethod
    def Open(spe_file):
        return _fallback_open_dataset(spe_file)


if gdal is None:
    gdal = _GdalShim()

gdal.UseExceptions()


def _parse_envi_header(hdr_file):
    header = {}
    current_key = None
    current_value = []

    with open(hdr_file, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line == "ENVI":
                continue
            if current_key is not None:
                current_value.append(line)
                if "}" in line:
                    header[current_key] = " ".join(current_value).strip()
                    current_key = None
                    current_value = []
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if value.startswith("{") and "}" not in value:
                current_key = key.lower()
                current_value = [value]
            else:
                header[key.lower()] = value

    return header


def _fallback_open_dataset(spe_file):
    root, _ = os.path.splitext(spe_file)
    hdr_file = root + ".hdr"
    if not os.path.exists(hdr_file):
        raise FileNotFoundError(f"Missing ENVI header for {spe_file}: {hdr_file}")
    return _FallbackEnviDataset(spe_file, _parse_envi_header(hdr_file))


def trans(data):
    if data.ndim == 2:
        data = np.fliplr(data)
        data = np.rot90(data, k=1)
    elif data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))
        data = np.fliplr(data)
        data = np.rot90(data, k=1)
        data = np.transpose(data, (2, 0, 1))
    return data


def getOnePicture(dataset, band_index):
    return dataset.GetRasterBand(band_index).ReadAsArray()


def read_data(spe_file):
    try:
        return gdal.Open(spe_file)
    except RuntimeError as exc:
        print(f"[WARN] Failed to open {spe_file}: {exc}")
        raise


def snv_normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std if std != 0 else x * 0.0


def _to_bytes(secret):
    if isinstance(secret, bytes):
        return secret
    if isinstance(secret, str):
        return secret.encode("utf-8")
    raise TypeError("secret must be str or bytes")


def _derive_key(secret, salt, iterations, length=32):
    return hashlib.pbkdf2_hmac(
        "sha256",
        _to_bytes(secret),
        salt,
        iterations,
        dklen=length,
    )


def _derive_legacy_keys(secret, salt, iterations):
    material = _derive_key(secret, salt, iterations, length=64)
    return material[:32], material[32:]


def _keystream(enc_key, nonce, size):
    blocks = []
    counter = 0
    while len(blocks) * 32 < size:
        counter_bytes = counter.to_bytes(8, "big")
        blocks.append(hmac.new(enc_key, nonce + counter_bytes, hashlib.sha256).digest())
        counter += 1
    return b"".join(blocks)[:size]


def _xor_bytes(data, mask):
    return bytes(left ^ right for left, right in zip(data, mask))


def _encrypt_bytes_legacy(plaintext, secret, iterations):
    salt = secrets.token_bytes(SALT_LEN)
    nonce = secrets.token_bytes(LEGACY_NONCE_LEN)
    enc_key, mac_key = _derive_legacy_keys(secret, salt, iterations)
    plaintext_bytes = bytes(plaintext)
    ciphertext = _xor_bytes(
        plaintext_bytes,
        _keystream(enc_key, nonce, len(plaintext_bytes)),
    )
    header = struct.pack(
        LEGACY_HEADER_FORMAT,
        LEGACY_MAGIC,
        LEGACY_VERSION,
        iterations,
        salt,
        nonce,
        len(plaintext_bytes),
    )
    tag = hmac.new(mac_key, header + ciphertext, hashlib.sha256).digest()
    return header + ciphertext + tag


def _decrypt_bytes_legacy(payload, secret):
    if len(payload) < LEGACY_HEADER_SIZE + LEGACY_TAG_LEN:
        raise ValueError("legacy encrypted payload is too short")

    header = payload[:LEGACY_HEADER_SIZE]
    ciphertext = payload[LEGACY_HEADER_SIZE:-LEGACY_TAG_LEN]
    tag = payload[-LEGACY_TAG_LEN:]
    magic, version, iterations, salt, nonce, plain_len = struct.unpack(
        LEGACY_HEADER_FORMAT,
        header,
    )
    if magic != LEGACY_MAGIC:
        raise ValueError("invalid legacy encrypted payload magic")
    if version != LEGACY_VERSION:
        raise ValueError(f"unsupported legacy encrypted payload version: {version}")

    enc_key, mac_key = _derive_legacy_keys(secret, salt, iterations)
    expected_tag = hmac.new(mac_key, header + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("legacy encrypted payload authentication failed")

    plaintext = _xor_bytes(ciphertext, _keystream(enc_key, nonce, len(ciphertext)))
    if len(plaintext) != plain_len:
        raise ValueError("legacy decrypted payload length mismatch")
    return plaintext


def encrypt_bytes(plaintext, secret, iterations=PBKDF2_ITERATIONS):
    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("plaintext must be bytes-like")
    if AESGCM is None:
        return _encrypt_bytes_legacy(plaintext, secret, iterations)

    plaintext_bytes = bytes(plaintext)
    salt = secrets.token_bytes(SALT_LEN)
    nonce = secrets.token_bytes(NONCE_LEN)
    key = _derive_key(secret, salt, iterations)
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        iterations,
        salt,
        nonce,
        len(plaintext_bytes),
    )
    ciphertext = AESGCM(key).encrypt(nonce, plaintext_bytes, header)
    return header + ciphertext


def decrypt_bytes(payload, secret):
    if len(payload) < 8:
        raise ValueError("encrypted payload is too short")
    magic = payload[:8]
    if magic == LEGACY_MAGIC:
        return _decrypt_bytes_legacy(payload, secret)
    if magic != MAGIC:
        raise ValueError("invalid encrypted payload magic")
    if AESGCM is None:
        raise ValueError("AES-GCM payload requires cryptography to be installed")
    if len(payload) < HEADER_SIZE + 16:
        raise ValueError("encrypted payload is too short")

    header = payload[:HEADER_SIZE]
    ciphertext = payload[HEADER_SIZE:]
    magic, version, iterations, salt, nonce, plain_len = struct.unpack(
        HEADER_FORMAT,
        header,
    )
    if version != VERSION:
        raise ValueError(f"unsupported encrypted payload version: {version}")

    key = _derive_key(secret, salt, iterations)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, header)
    if len(plaintext) != plain_len:
        raise ValueError("decrypted payload length mismatch")
    return plaintext


def encrypt_file(input_path, output_path, secret, iterations=PBKDF2_ITERATIONS):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        encrypt_bytes(input_path.read_bytes(), secret, iterations=iterations)
    )
    return output_path


def decrypt_file(input_path, output_path, secret):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decrypt_bytes(input_path.read_bytes(), secret))
    return output_path


def _resolve_secret(key=None, key_env=DEFAULT_KEY_ENV):
    if key is not None:
        return key
    if key_env is not None:
        secret = os.getenv(key_env)
        if secret:
            return secret
        raise ValueError(f"Environment variable {key_env} is not set.")
    raise ValueError("Provide a key or key_env to encrypt model files.")


def _iter_encrypt_targets(input_dir, recursive):
    walker = input_dir.rglob if recursive else input_dir.glob
    for pattern in TARGET_PATTERNS:
        for path in walker(pattern):
            if path.is_file() and path.suffix.lower() != ".enc":
                yield path


def encrypt_model_directory(
    input_dir,
    output_dir,
    key=None,
    key_env=DEFAULT_KEY_ENV,
    recursive=False,
    overwrite=False,
    manifest_path=None,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    secret = _resolve_secret(key=key, key_env=key_env)
    targets = sorted(set(_iter_encrypt_targets(input_dir, recursive)))
    if not targets:
        raise FileNotFoundError(
            f"No {TARGET_PATTERNS} files found under {input_dir} (recursive={recursive})."
        )

    manifest_entries = []
    for source_path in targets:
        relative = source_path.relative_to(input_dir)
        destination = output_dir / Path(f"{relative}.enc")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Encrypted output already exists: {destination}. Use overwrite=True to replace it."
            )
        encrypt_file(source_path, destination, secret)
        manifest_entries.append(
            {
                "source": str(source_path),
                "encrypted": str(destination),
                "size": source_path.stat().st_size,
            }
        )
        print(f"[ENC] {source_path} -> {destination}")

    if manifest_path is None:
        manifest_path = output_dir / "encryption_manifest.json"
    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "recursive": recursive,
                "files": manifest_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ENC] wrote manifest: {manifest_path}")
    return manifest_path


def _is_encrypted_path(path):
    return Path(path).suffix.lower() == ".enc"


def _resolve_model_key(model_key):
    if model_key is not None:
        return model_key
    env_value = os.getenv(MODEL_KEY_ENV)
    if env_value:
        return env_value
    return BUILTIN_MODEL_KEY


def _require_model_key(model_key):
    resolved = _resolve_model_key(model_key)
    if not resolved:
        raise ValueError(
            "Encrypted model detected but no key was provided. "
            f"Pass model_key=... or set {MODEL_KEY_ENV}."
        )
    return resolved


def _build_temperature_candidates(model_path):
    path = Path(model_path)
    if _is_encrypted_path(path):
        path = path.with_suffix("")
    base_prefix = path.with_suffix("")
    temp_plain = base_prefix.parent / f"{base_prefix.name}_temperature.txt"
    temp_encrypted = Path(f"{temp_plain}.enc")
    return temp_plain, temp_encrypted


def _load_temperature(model_path, model_key, default_temp):
    temp_plain, temp_encrypted = _build_temperature_candidates(model_path)
    model_name = Path(model_path).name

    if temp_encrypted.exists():
        try:
            secret = _require_model_key(model_key)
            temperature = float(
                decrypt_bytes(temp_encrypted.read_bytes(), secret).decode("utf-8").strip()
            )
            print(f"[INFO] Loaded encrypted temperature {temperature:.4f} for {model_name}")
            return temperature
        except Exception as exc:
            print(
                f"[WARN] Failed to read encrypted temperature file {temp_encrypted}: {exc}. "
                f"Using default {default_temp}."
            )
            return default_temp

    if temp_plain.exists():
        try:
            temperature = float(temp_plain.read_text(encoding="utf-8").strip())
            print(f"[INFO] Loaded temperature {temperature:.4f} for {model_name}")
            return temperature
        except Exception as exc:
            print(
                f"[WARN] Failed to read temperature file {temp_plain}: {exc}. "
                f"Using default {default_temp}."
            )
            return default_temp

    print(f"[INFO] No temperature file for {model_name}. Using default {default_temp}.")
    return default_temp


def load_model_with_temperature(pt_path, device="cpu", default_temp=1.0, model_key=None):
    if _is_encrypted_path(pt_path):
        secret = _require_model_key(model_key)
        model = torch.jit.load(
            io.BytesIO(decrypt_bytes(Path(pt_path).read_bytes(), secret)),
            map_location=device,
        )
        print(f"[INFO] Loaded encrypted model: {os.path.basename(pt_path)}")
    else:
        model = torch.jit.load(pt_path, map_location=device)
        print(f"[INFO] Loaded model: {os.path.basename(pt_path)}")

    model.eval()
    temperature = _load_temperature(pt_path, model_key, default_temp)
    return model, temperature


def _normalize_thresholds(thresholds):
    if thresholds is None:
        return [0.5]
    if isinstance(thresholds, (int, float)):
        return [float(thresholds)]
    return [float(value) for value in thresholds]


def _parse_center_position_text(text):
    payload = text.strip()
    if payload.startswith("center_position"):
        _, _, payload = payload.partition("=")
        payload = payload.strip()
    try:
        return ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return json.loads(payload)


def _resolve_center_position(center_position):
    if center_position is None:
        return None
    if isinstance(center_position, (list, tuple)):
        return center_position
    if isinstance(center_position, str):
        stripped = center_position.strip()
        if stripped.startswith(("center_position", "[", "(", "{")):
            return _parse_center_position_text(stripped)
        center_path = Path(center_position)
        if center_path.exists():
            return json.loads(center_path.read_text(encoding="utf-8"))
    raise TypeError(
        "center_position must be None, a sequence, JSON text, or a JSON file path."
    )


def _flatten_centers(center_position):
    if center_position is None:
        return [(x, y) for row in ADJUSTED_IMAGES_CENTER for (x, y) in row]

    flattened = []
    for item in center_position:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
            flattened.extend((int(x), int(y)) for x, y in item)
        else:
            x, y = item
            flattened.append((int(x), int(y)))
    return flattened


def _sort_centers(centers):
    return sorted(
        ((int(x), int(y)) for x, y in centers),
        key=lambda item: (item[1], -item[0]),
    )


def _infer_grid_shape(center_position):
    if center_position is None:
        return len(ADJUSTED_IMAGES_CENTER), len(ADJUSTED_IMAGES_CENTER[0])

    if (
        isinstance(center_position, (list, tuple))
        and center_position
        and isinstance(center_position[0], (list, tuple))
        and center_position[0]
        and isinstance(center_position[0][0], (list, tuple))
    ):
        row_lengths = [len(row) for row in center_position]
        if not row_lengths or any(length != row_lengths[0] for length in row_lengths):
            raise ValueError(
                f"center_position must describe a rectangular grid, got row lengths {row_lengths}."
            )
        return len(row_lengths), row_lengths[0]

    sorted_centers = _sort_centers(_flatten_centers(center_position))
    if not sorted_centers:
        raise ValueError("center_position must contain at least one point.")

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
        raise ValueError(
            "Could not infer a rectangular grid from center_position. "
            f"Grouped row lengths: {row_lengths}."
        )
    return len(row_groups), row_lengths[0]


def _collect_model_paths(model_dir):
    model_dir_path = Path(model_dir)
    encrypted_paths = sorted(str(path) for path in model_dir_path.glob("*.pt.enc"))
    if encrypted_paths:
        return encrypted_paths

    plain_paths = sorted(str(path) for path in model_dir_path.glob("*.pt"))
    if plain_paths:
        return plain_paths

    candidate_dirs = sorted(
        str(path)
        for path in model_dir_path.iterdir()
        if path.is_dir() and (any(path.glob("*.pt.enc")) or any(path.glob("*.pt")))
    )
    if candidate_dirs:
        joined = ", ".join(candidate_dirs)
        raise FileNotFoundError(
            f"No .pt or .pt.enc files found directly in {model_dir}. "
            f"Choose one of these model-set directories instead: {joined}"
        )
    raise FileNotFoundError(f"No .pt or .pt.enc files found under {model_dir}.")


def JsonSegEgg(path, center_position=None):
    dataset = read_data(path)
    data = cv2.convertScaleAbs(trans(getOnePicture(dataset, 150)))
    centers = _sort_centers(_flatten_centers(center_position))

    sorted_area_info = [
        (max(0, x - 10), max(0, y - 10), min(310, x + 10), min(480, y + 10))
        for x, y in centers
    ]

    data_all_bands = np.array(
        [dataset.GetRasterBand(band).ReadAsArray() for band in range(1, dataset.RasterCount + 1)]
    )

    egg_data_list = []
    for x1, y1, x2, y2 in sorted_area_info:
        window = data_all_bands[:, x1:x2, y1:y2]
        sliced = np.transpose(window, (1, 2, 0))[:, :, 100:250]
        egg_data_list.append(sliced.reshape(-1, sliced.shape[-1]).tolist())
    return egg_data_list


def _prepare_tensor(data, batch_size):
    spectral_data = np.asarray(data, dtype=float)
    snv_data = np.apply_along_axis(snv_normalize, 1, spectral_data)
    sg_data = savgol_filter(snv_data, window_length=9, polyorder=2, deriv=1, axis=1)
    pooled = np.median(sg_data, axis=0).reshape(1, -1)
    pooled = pooled - pooled.mean()
    return torch.tensor(pooled, dtype=torch.float32).unsqueeze(0).reshape(
        batch_size, 1, 1, 150
    )


def pred_test(models_with_temp, tensor, device, num_f, num_m, index, threshold, num_cols):
    with torch.no_grad():
        inputs = tensor.to(device)
        all_preds = []

        for model, temperature in models_with_temp:
            outputs = model(inputs) / temperature
            probabilities = torch.softmax(outputs, dim=1)
            female_prob = probabilities[:, 0]
            all_preds.append(1 - (female_prob >= threshold).long())

        predicted = torch.min(torch.stack(all_preds), dim=0)[0]
        num_f += (predicted == 0).sum().item()
        num_m += (predicted == 1).sum().item()
        egg_row = (index // num_cols) + 1
        egg_col = (index % num_cols) + 1
        egg_index = f"{egg_row}-{egg_col}"
        predict = {egg_index: int(predicted.item()) + 1}

    return predict, num_f, num_m


def plate_cl(data_list, model_paths, batch_size, thresholds, grid_shape, model_key=None):
    device = torch.device("cpu")
    models_with_temp = [
        load_model_with_temperature(path, device, model_key=model_key)
        for path in model_paths
    ]
    _, num_cols = grid_shape
    results = []
    data_array = np.array(data_list, dtype=object)

    for threshold in _normalize_thresholds(thresholds):
        predict_dict = {}
        num_f = 0
        num_m = 0
        print(f"[INFO] Running threshold={threshold}")

        for index in range(len(data_array)):
            tensor = _prepare_tensor(data_array[index], batch_size)
            predict, num_f, num_m = pred_test(
                models_with_temp,
                tensor,
                device,
                num_f,
                num_m,
                index,
                threshold,
                num_cols,
            )
            predict_dict.update(predict)

        results.append(
            {
                "threshold": threshold,
                "predict": predict_dict,
                "num_f": num_f,
                "num_m": num_m,
            }
        )
        print(f"[INFO] Female: {num_f}, Male: {num_m}")
    return results


def writeJsonToCsharp(mem_key, predict, num_f, num_m):
    return json.dumps(
        {
            "baseInfo": {"palletCode": mem_key},
            "statistics": {"female": num_f, "male": num_m},
            "rawData": predict,
        }
    )


def buildSpeVPath(spe_binary_data, hdr_str, mem_key):
    if hasattr(gdal, "FileFromMemBuffer"):
        spe_path = f"/vsimem/{mem_key}.spe"
        hdr_path = f"/vsimem/{mem_key}.hdr"
        gdal.FileFromMemBuffer(spe_path, bytes(spe_binary_data))
        gdal.FileFromMemBuffer(hdr_path, bytes(hdr_str))
        return spe_path, hdr_path

    temp_root = Path(__file__).resolve().parent / "_tmp_vsis"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{mem_key}_", dir=str(temp_root)))
    spe_path = temp_dir / f"{mem_key}.spe"
    hdr_path = temp_dir / f"{mem_key}.hdr"
    spe_path.write_bytes(bytes(spe_binary_data))
    hdr_path.write_bytes(bytes(hdr_str))
    return str(spe_path), str(hdr_path)


def freeVPath(spe_path, hdr_path):
    if isinstance(spe_path, str) and spe_path.startswith("/vsimem/") and hasattr(gdal, "Unlink"):
        gdal.Unlink(spe_path)
        gdal.Unlink(hdr_path)
        return

    for path in (spe_path, hdr_path):
        try:
            Path(path).unlink(missing_ok=True)
        except TypeError:
            if Path(path).exists():
                Path(path).unlink()
    try:
        Path(spe_path).parent.rmdir()
    except OSError:
        pass


def identifyEggs(
    spe_bin,
    hdr_str,
    model_dir,
    mem_key=11,
    thresholds=None,
    center_position=None,
    model_key=None,
):
    model_paths = _collect_model_paths(model_dir)
    resolved_centers = _resolve_center_position(center_position)
    grid_shape = _infer_grid_shape(resolved_centers)
    v_spe_path, v_hdr_path = buildSpeVPath(spe_bin, hdr_str, mem_key)

    print(f"[INFO] Processing pallet {mem_key}")
    t_seg_start = time.perf_counter()
    test_list = JsonSegEgg(v_spe_path, resolved_centers)
    t_seg_end = time.perf_counter()

    t_pred_start = time.perf_counter()
    results = plate_cl(
        test_list,
        model_paths,
        1,
        thresholds,
        grid_shape,
        model_key=model_key,
    )
    t_pred_end = time.perf_counter()

    print(f"[INFO] Prediction time: {t_pred_end - t_pred_start:.4f}s")
    print(f"[INFO] Segmentation time: {t_seg_end - t_seg_start:.4f}s")

    payload = None
    for result in results:
        payload = writeJsonToCsharp(
            mem_key,
            result["predict"],
            result["num_f"],
            result["num_m"],
        )
        print(f"[INFO] threshold={result['threshold']} json={payload}")

    freeVPath(v_spe_path, v_hdr_path)
    return payload


def test():
    script_dir = Path(__file__).resolve().parent
    model_root = script_dir.parent / "Model"
    if not model_root.exists():
        print(f"[INFO] Example model root does not exist: {model_root}")
        return

    model_sets = sorted(
        path for path in model_root.iterdir() if path.is_dir() and any(path.glob("*.pt"))
    )
    if not model_sets:
        print(f"[INFO] No model-set directory found under {model_root}")
        return

    print(f"[INFO] Example model_dir: {model_sets[0]}")
    print("[INFO] Use run_infer_test.py for a full smoke test.")


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Single-file Egg inference module with built-in model encryption."
    )
    subparsers = parser.add_subparsers(dest="command")

    encrypt_parser = subparsers.add_parser(
        "encrypt-models",
        help="Encrypt TorchScript model files and companion temperature files.",
    )
    encrypt_parser.add_argument("--input-dir", type=Path, required=True)
    encrypt_parser.add_argument("--output-dir", type=Path, required=True)
    encrypt_parser.add_argument("--key", default=None)
    encrypt_parser.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    encrypt_parser.add_argument("--recursive", action="store_true")
    encrypt_parser.add_argument("--overwrite", action="store_true")
    encrypt_parser.add_argument("--manifest", type=Path, default=None)

    return parser


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "encrypt-models":
        encrypt_model_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            key=args.key,
            key_env=args.key_env,
            recursive=args.recursive,
            overwrite=args.overwrite,
            manifest_path=args.manifest,
        )
        return

    test()


if __name__ == "__main__":
    main()
