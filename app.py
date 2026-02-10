import os
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
BASE_VIDEO_PATH = os.path.join(ASSETS_DIR, "base.mp4")

# Advanced watermark parameters
BIT_FRAMES = int(os.environ.get("BIT_FRAMES", "8"))
CHIPS_PER_BIT = int(os.environ.get("CHIPS_PER_BIT", "4"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "28"))
DELTA = int(os.environ.get("DELTA", "7"))
DELTA_DC = int(os.environ.get("DELTA_DC", "3"))
DC_WEIGHT = float(os.environ.get("DC_WEIGHT", "0.60"))
DCT_WEIGHT = float(os.environ.get("DCT_WEIGHT", "0.55"))
DCT_QIM_DELTA = float(os.environ.get("DCT_QIM_DELTA", "14"))
PATTERN_BLOCK = int(os.environ.get("PATTERN_BLOCK", "7"))
BASE_FPS_FALLBACK = float(os.environ.get("BASE_FPS", "24"))
PRN_SEED = int(os.environ.get("PRN_SEED", "915131"))
MAX_PAYLOAD_BYTES = int(os.environ.get("MAX_PAYLOAD_BYTES", "128"))
REPEAT_PACKET = os.environ.get("REPEAT_PACKET", "1").strip().lower() not in {"0", "false", "no"}
PACKET_GAP_BITS = int(os.environ.get("PACKET_GAP_BITS", "0"))
INTERLEAVER_DEPTH = max(1, int(os.environ.get("INTERLEAVER_DEPTH", "8")))
PILOT_BITS = [int(bit) for bit in os.environ.get("PILOT_BITS", "101011001101")]
PILOT_EVERY_S = float(os.environ.get("PILOT_EVERY_S", "1.0"))

# Geometric registration markers
REG_MARKERS = os.environ.get("REG_MARKERS", "1").strip().lower() not in {"0", "false", "no"}
REG_MARKER_DELTA = int(os.environ.get("REG_MARKER_DELTA", "12"))
REG_MARKER_SIZE = int(os.environ.get("REG_MARKER_SIZE", "22"))
REG_MARKER_BLOCK = int(os.environ.get("REG_MARKER_BLOCK", "3"))

# Visible fallback micro-strip (highly robust to recording pipelines)
VISIBLE_STRIP = os.environ.get("VISIBLE_STRIP", "1").strip().lower() not in {"0", "false", "no"}
STRIP_SYNC_BITS = [int(bit) for bit in os.environ.get("STRIP_SYNC_BITS", "111000101011110010100111")]
STRIP_CAP_BITS = int(os.environ.get("STRIP_CAP_BITS", "512"))
STRIP_MARGIN = int(os.environ.get("STRIP_MARGIN", "8"))
STRIP_HEIGHT = int(os.environ.get("STRIP_HEIGHT", "10"))
STRIP_ALPHA = float(os.environ.get("STRIP_ALPHA", "0.80"))

# Capture constraints (decode quality gate hints)
MIN_CAPTURE_WIDTH = int(os.environ.get("MIN_CAPTURE_WIDTH", "960"))
MIN_CAPTURE_HEIGHT = int(os.environ.get("MIN_CAPTURE_HEIGHT", "540"))
MIN_CAPTURE_FPS = float(os.environ.get("MIN_CAPTURE_FPS", "20.0"))
MIN_CAPTURE_BITRATE_KBPS = int(os.environ.get("MIN_CAPTURE_BITRATE_KBPS", "1500"))

PREAMBLE_BITS = [int(bit) for bit in "11001011100101101001110010110100"]
CELL_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (0.18, 0.22),
    (0.50, 0.22),
    (0.82, 0.22),
    (0.18, 0.50),
    (0.50, 0.50),
    (0.82, 0.50),
    (0.18, 0.78),
    (0.50, 0.78),
    (0.82, 0.78),
)
REG_MARKER_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (0.08, 0.08),
    (0.92, 0.08),
    (0.08, 0.92),
    (0.92, 0.92),
)
REG_MARKER_POLARITY: Tuple[int, ...] = (1, -1, -1, 1)

# Legacy decoder compatibility (previous algorithm)
LEGACY_PATCH_SIZE = int(os.environ.get("LEGACY_PATCH_SIZE", "32"))
LEGACY_PATCH_X = int(os.environ.get("LEGACY_PATCH_X", "50"))
LEGACY_PATCH_Y = int(os.environ.get("LEGACY_PATCH_Y", "50"))
LEGACY_SYNC_PREAMBLE = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]

app = Flask(__name__)
CHECKERBOARD_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
REG_MARKER_CACHE: Dict[int, np.ndarray] = {}


def ensure_dirs() -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_placeholder_video(
    path: str, duration_s: int = 6, fps: int = 30, size: Tuple[int, int] = (640, 360)
) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    frames = duration_s * fps

    for i in range(frames):
        t = i / max(frames - 1, 1)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.uint8(40 + 80 * t)
        frame[:, :, 1] = np.uint8(60 + 120 * (1 - t))
        frame[:, :, 2] = np.uint8(80 + 60 * np.sin(t * np.pi))
        cv2.putText(
            frame,
            "BASE VIDEO",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def get_base_video_fps() -> float:
    if os.path.exists(BASE_VIDEO_PATH):
        cap = cv2.VideoCapture(BASE_VIDEO_PATH)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if 1.0 <= fps <= 240.0:
                return fps
    return BASE_FPS_FALLBACK


def get_video_shape(path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def get_base_video_shape() -> Tuple[int, int]:
    if not os.path.exists(BASE_VIDEO_PATH):
        return 0, 0
    return get_video_shape(BASE_VIDEO_PATH)


def capture_constraints_report(path: str) -> Dict[str, object]:
    report: Dict[str, object] = {
        "ok": True,
        "violations": [],
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration_s": 0.0,
        "bitrate_kbps": 0.0,
    }
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        report["ok"] = False
        report["violations"] = ["unable_to_open_video"]
        return report

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    duration_s = (frames / fps) if fps > 0 else 0.0
    size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    bitrate_kbps = ((size_bytes * 8) / max(duration_s, 0.001)) / 1000.0 if duration_s > 0 else 0.0
    violations: List[str] = []

    if width < MIN_CAPTURE_WIDTH or height < MIN_CAPTURE_HEIGHT:
        violations.append("resolution_too_low")
    if fps < MIN_CAPTURE_FPS:
        violations.append("fps_too_low")
    if bitrate_kbps < MIN_CAPTURE_BITRATE_KBPS:
        violations.append("bitrate_too_low")

    report.update(
        {
            "ok": len(violations) == 0,
            "violations": violations,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_s,
            "bitrate_kbps": bitrate_kbps,
        }
    )
    return report


def bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    out = bytearray()
    usable = len(bits) - (len(bits) % 8)
    for i in range(0, usable, 8):
        value = 0
        for bit in bits[i : i + 8]:
            value = (value << 1) | (bit & 1)
        out.append(value)
    return bytes(out)


def repeat_bits(bits: Sequence[int], repeat_factor: int) -> List[int]:
    if repeat_factor <= 1:
        return [bit & 1 for bit in bits]
    out: List[int] = []
    for bit in bits:
        out.extend([(bit & 1)] * repeat_factor)
    return out


def collapse_repeated_bits(bits: Sequence[int], repeat_factor: int) -> List[int]:
    if repeat_factor <= 1:
        return [bit & 1 for bit in bits]
    usable = len(bits) - (len(bits) % repeat_factor)
    out: List[int] = []
    for i in range(0, usable, repeat_factor):
        chunk = bits[i : i + repeat_factor]
        ones = sum(1 for bit in chunk if bit)
        out.append(1 if ones * 2 >= repeat_factor else 0)
    return out


def interleave_bits(bits: Sequence[int], depth: int) -> List[int]:
    if depth <= 1 or not bits:
        return [bit & 1 for bit in bits]

    rows = max(depth, 1)
    cols = (len(bits) + rows - 1) // rows
    grid = [[-1] * cols for _ in range(rows)]
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= len(bits):
                break
            grid[row][col] = bits[idx] & 1
            idx += 1

    out: List[int] = []
    for col in range(cols):
        for row in range(rows):
            value = grid[row][col]
            if value >= 0:
                out.append(value)
    return out


def deinterleave_bits(bits: Sequence[int], depth: int) -> List[int]:
    if depth <= 1 or not bits:
        return [bit & 1 for bit in bits]

    rows = max(depth, 1)
    cols = (len(bits) + rows - 1) // rows
    row_lengths = [0] * rows
    for row in range(rows):
        start = row * cols
        remaining = max(len(bits) - start, 0)
        row_lengths[row] = min(cols, remaining)
    grid = [[-1] * row_lengths[row] for row in range(rows)]

    idx = 0
    for col in range(cols):
        for row in range(rows):
            if col >= row_lengths[row]:
                continue
            if idx >= len(bits):
                break
            grid[row][col] = bits[idx] & 1
            idx += 1

    out: List[int] = []
    for row in range(rows):
        for col in range(row_lengths[row]):
            value = grid[row][col]
            if value >= 0:
                out.append(value)
    return out


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    poly = 0x1021
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc32_ieee(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def hamming74_encode(bits: Sequence[int]) -> List[int]:
    encoded: List[int] = []
    padded = list(bits)
    remainder = len(padded) % 4
    if remainder:
        padded.extend([0] * (4 - remainder))

    for i in range(0, len(padded), 4):
        d1, d2, d3, d4 = (padded[i + j] & 1 for j in range(4))
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        encoded.extend([p1, p2, d1, p3, d2, d3, d4])
    return encoded


def hamming74_decode(
    bits: Sequence[int], max_data_bits: Optional[int] = None
) -> Tuple[List[int], int]:
    decoded: List[int] = []
    corrected = 0
    usable = len(bits) - (len(bits) % 7)

    for i in range(0, usable, 7):
        block = [bits[i + j] & 1 for j in range(7)]
        s1 = block[0] ^ block[2] ^ block[4] ^ block[6]
        s2 = block[1] ^ block[2] ^ block[5] ^ block[6]
        s3 = block[3] ^ block[4] ^ block[5] ^ block[6]
        error_pos = s1 | (s2 << 1) | (s3 << 2)

        if 1 <= error_pos <= 7:
            block[error_pos - 1] ^= 1
            corrected += 1

        decoded.extend([block[2], block[4], block[5], block[6]])
        if max_data_bits is not None and len(decoded) >= max_data_bits:
            return decoded[:max_data_bits], corrected

    if max_data_bits is not None:
        decoded = decoded[:max_data_bits]
    return decoded, corrected


def estimate_max_payload_bytes(bit_capacity: int) -> int:
    if bit_capacity <= len(PREAMBLE_BITS):
        return 0
    ecc_bits = bit_capacity - len(PREAMBLE_BITS)
    raw_bits = (ecc_bits // 7) * 4
    raw_bytes = raw_bits // 8
    return max(raw_bytes - 7, 0)  # 3 bytes header + 4 bytes CRC32


PACKET_MAGIC = 0x57  # 'W'
PACKET_VERSION = 0x03


def build_packet_bits(text: str, bit_capacity_hint: Optional[int] = None) -> List[int]:
    payload = text.encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"ID too long ({len(payload)} bytes). Max supported is {MAX_PAYLOAD_BYTES} bytes."
        )
    if len(payload) > 255:
        raise ValueError("ID too long for packet format. Max 255 bytes.")

    repeat_factor = 1
    if bit_capacity_hint is not None:
        for candidate in (2, 1):
            flags = (PACKET_VERSION << 4) | (candidate & 0x03)
            if INTERLEAVER_DEPTH > 1:
                flags |= 0x04
            header = bytes((PACKET_MAGIC, flags, len(payload)))
            crc = crc32_ieee(header + payload).to_bytes(4, "big")
            raw_bits = bytes_to_bits(header + payload + crc)
            ecc_bits = hamming74_encode(raw_bits)
            if INTERLEAVER_DEPTH > 1:
                ecc_bits = interleave_bits(ecc_bits, INTERLEAVER_DEPTH)
            sized = repeat_bits(ecc_bits, candidate)
            if len(PREAMBLE_BITS) + len(sized) <= bit_capacity_hint:
                repeat_factor = candidate
                break

    flags = (PACKET_VERSION << 4) | (repeat_factor & 0x03)
    if INTERLEAVER_DEPTH > 1:
        flags |= 0x04
    header = bytes((PACKET_MAGIC, flags, len(payload)))
    crc = crc32_ieee(header + payload).to_bytes(4, "big")
    raw_bits = bytes_to_bits(header + payload + crc)
    ecc_bits = hamming74_encode(raw_bits)
    if INTERLEAVER_DEPTH > 1:
        ecc_bits = interleave_bits(ecc_bits, INTERLEAVER_DEPTH)
    ecc_bits = repeat_bits(ecc_bits, repeat_factor)
    return PREAMBLE_BITS + ecc_bits


def decode_packet_bits_v3(bit_stream: Sequence[int]) -> Tuple[Optional[str], Dict[str, int]]:
    best_info: Dict[str, int] = {"payload_len": -1, "corrected": 0, "crc_ok": 0}
    header_ecc_bits = 42  # 3 raw bytes => 24 bits => 42 ECC bits

    for repeat_factor in (2, 1):
        collapsed = collapse_repeated_bits(bit_stream, repeat_factor)

        for deinterleave in (True, False):
            stream = (
                deinterleave_bits(collapsed, INTERLEAVER_DEPTH)
                if deinterleave and INTERLEAVER_DEPTH > 1
                else list(collapsed)
            )
            if len(stream) < header_ecc_bits:
                continue

            header_bits, corrected_header = hamming74_decode(
                stream[:header_ecc_bits], max_data_bits=24
            )
            header_bytes = bits_to_bytes(header_bits)
            if len(header_bytes) < 3:
                continue

            magic, flags, payload_len = header_bytes[:3]
            if magic != PACKET_MAGIC:
                continue

            version = (flags >> 4) & 0x0F
            encoded_repeat = flags & 0x03
            encoded_interleave = 1 if (flags & 0x04) else 0
            if version != PACKET_VERSION:
                continue
            if encoded_repeat and encoded_repeat != repeat_factor:
                continue
            if encoded_interleave != (1 if deinterleave and INTERLEAVER_DEPTH > 1 else 0):
                continue
            if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
                continue

            total_raw_bytes = 3 + payload_len + 4
            total_raw_bits = total_raw_bytes * 8
            total_ecc_bits = (total_raw_bits // 4) * 7
            if len(stream) < total_ecc_bits:
                continue

            raw_bits, corrected_payload = hamming74_decode(
                stream[:total_ecc_bits], max_data_bits=total_raw_bits
            )
            raw_bytes = bits_to_bytes(raw_bits)
            if len(raw_bytes) < total_raw_bytes:
                continue

            if raw_bytes[0] != PACKET_MAGIC:
                continue
            if raw_bytes[2] != payload_len:
                continue

            payload = raw_bytes[3 : 3 + payload_len]
            crc_read = int.from_bytes(raw_bytes[3 + payload_len : 3 + payload_len + 4], "big")
            crc_calc = crc32_ieee(raw_bytes[: 3 + payload_len])
            if crc_read != crc_calc:
                info = {
                    "payload_len": payload_len,
                    "corrected": corrected_header + corrected_payload,
                    "crc_ok": 0,
                }
                if info["corrected"] < best_info.get("corrected", 1_000_000):
                    best_info = info
                continue

            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue

            return (
                text,
                {
                    "payload_len": payload_len,
                    "corrected": corrected_header + corrected_payload,
                    "crc_ok": 1,
                },
            )

    return None, best_info


def decode_packet_bits_v2(bit_stream: Sequence[int]) -> Tuple[Optional[str], Dict[str, int]]:
    info: Dict[str, int] = {"payload_len": -1, "corrected": 0, "crc_ok": 0}

    header_ecc_bits = 28  # 16 raw bits -> 4 nibbles -> 28 ECC bits
    if len(bit_stream) < header_ecc_bits:
        return None, info

    header_bits, corrected_header = hamming74_decode(
        bit_stream[:header_ecc_bits], max_data_bits=16
    )
    info["corrected"] += corrected_header
    header_bytes = bits_to_bytes(header_bits)
    if len(header_bytes) < 2:
        return None, info

    payload_len = int.from_bytes(header_bytes[:2], "big")
    info["payload_len"] = payload_len
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        return None, info

    total_raw_bytes = 2 + payload_len + 2
    total_raw_bits = total_raw_bytes * 8
    total_ecc_bits = (total_raw_bits // 4) * 7
    if len(bit_stream) < total_ecc_bits:
        return None, info

    raw_bits, corrected_payload = hamming74_decode(
        bit_stream[:total_ecc_bits], max_data_bits=total_raw_bits
    )
    info["corrected"] += corrected_payload

    raw_bytes = bits_to_bytes(raw_bits)
    if len(raw_bytes) < total_raw_bytes:
        return None, info

    parsed_len = int.from_bytes(raw_bytes[:2], "big")
    if parsed_len != payload_len:
        return None, info

    payload = raw_bytes[2 : 2 + payload_len]
    crc_read = int.from_bytes(raw_bytes[2 + payload_len : 2 + payload_len + 2], "big")
    if crc_read != crc16_ccitt(payload):
        return None, info

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return None, info

    info["crc_ok"] = 1
    return text, info


def decode_packet_bits(bit_stream: Sequence[int]) -> Tuple[Optional[str], Dict[str, int]]:
    # Try modern packet format first.
    text, info = decode_packet_bits_v3(bit_stream)
    if text is not None:
        return text, info

    # Backward compatibility with previous packet format.
    text_v2, info_v2 = decode_packet_bits_v2(bit_stream)
    if text_v2 is not None:
        return text_v2, info_v2

    # Tolerate older streams that may have used interleaving.
    if INTERLEAVER_DEPTH > 1:
        deint = deinterleave_bits(bit_stream, INTERLEAVER_DEPTH)
        text_deint, info_deint = decode_packet_bits_v2(deint)
        if text_deint is not None:
            return text_deint, info_deint

    if info_v2.get("payload_len", -1) >= 0:
        return None, info_v2
    return None, info


def prn_sign(bit_index: int, chip_index: int, cell_index: int) -> int:
    x = PRN_SEED & 0xFFFFFFFF
    x ^= ((bit_index + 1) * 0x9E3779B1) & 0xFFFFFFFF
    x ^= ((chip_index + 1) * 0x85EBCA77) & 0xFFFFFFFF
    x ^= ((cell_index + 1) * 0xC2B2AE3D) & 0xFFFFFFFF
    x &= 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    return 1 if (x & 1) else -1


def resolve_cells(frame: np.ndarray, patch_size: Optional[int] = None) -> List[Tuple[int, int, int]]:
    height, width = frame.shape[:2]
    target_patch = patch_size if patch_size is not None else PATCH_SIZE
    size = min(max(target_patch, 1), max(width // 10, 8), max(height // 10, 8))
    cells: List[Tuple[int, int, int]] = []

    for cx, cy in CELL_ANCHORS:
        x0 = int(round(cx * (width - 1) - size / 2))
        y0 = int(round(cy * (height - 1) - size / 2))
        x0 = max(0, min(width - size, x0))
        y0 = max(0, min(height - size, y0))
        cells.append((x0, y0, size))
    return cells


def checkerboard_pattern(size: int, block_size: Optional[int] = None) -> np.ndarray:
    block = max(block_size if block_size is not None else PATTERN_BLOCK, 1)
    key = (size, block)
    cached = CHECKERBOARD_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = np.indices((size, size))
    pattern = ((((xx // block) + (yy // block)) & 1) * 2 - 1).astype(np.int16)
    CHECKERBOARD_CACHE[key] = pattern
    return pattern


def registration_marker_pattern(size: int) -> np.ndarray:
    cached = REG_MARKER_CACHE.get(size)
    if cached is not None:
        return cached
    block = max(REG_MARKER_BLOCK, 1)
    yy, xx = np.indices((size, size))
    checker = ((((xx // block) + (yy // block)) & 1) * 2 - 1).astype(np.int16)
    radial = ((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2) ** 0.5
    radius = max(size * 0.42, 1.0)
    ring = np.where(radial < radius, 1, -1).astype(np.int16)
    pattern = checker * ring
    REG_MARKER_CACHE[size] = pattern
    return pattern


def apply_dct_qim_watermark(patch: np.ndarray, sign: int) -> np.ndarray:
    if patch.shape[0] < 6 or patch.shape[1] < 6 or DCT_WEIGHT <= 0 or DCT_QIM_DELTA <= 0:
        return patch
    patch_u8 = patch.astype(np.uint8)
    ycc = cv2.cvtColor(patch_u8, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0] - 128.0
    dct = cv2.dct(y)

    c1 = float(dct[2, 3])
    c2 = float(dct[3, 2])
    target = float(sign) * DCT_QIM_DELTA
    delta = target - (c1 - c2)
    dct[2, 3] = c1 + (delta * 0.5)
    dct[3, 2] = c2 - (delta * 0.5)

    y2 = cv2.idct(dct) + 128.0
    ycc[:, :, 0] = np.clip(y2, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.int16)


def apply_registration_markers(frame: np.ndarray) -> np.ndarray:
    if not REG_MARKERS:
        return frame

    height, width = frame.shape[:2]
    size = min(REG_MARKER_SIZE, max(width // 12, 10), max(height // 12, 10))
    size = max(size, 8)
    pattern = registration_marker_pattern(size)

    for (cx, cy), polarity in zip(REG_MARKER_ANCHORS, REG_MARKER_POLARITY):
        x0 = int(round(cx * (width - 1) - size / 2))
        y0 = int(round(cy * (height - 1) - size / 2))
        x0 = max(0, min(width - size, x0))
        y0 = max(0, min(height - size, y0))

        patch = frame[y0 : y0 + size, x0 : x0 + size].astype(np.int16)
        modulation = pattern * polarity * REG_MARKER_DELTA
        patch[:, :, 0] = np.clip(patch[:, :, 0] + modulation, 0, 255)
        patch[:, :, 2] = np.clip(patch[:, :, 2] - modulation, 0, 255)
        frame[y0 : y0 + size, x0 : x0 + size] = patch.astype(np.uint8)

    return frame


def apply_spread_watermark(
    frame: np.ndarray, bit_index: int, chip_index: int, bit_value: int
) -> np.ndarray:
    cells = resolve_cells(frame)
    bit_polarity = 1 if bit_value == 1 else -1

    for cell_index, (x0, y0, size) in enumerate(cells):
        sign = prn_sign(bit_index, chip_index, cell_index) * bit_polarity
        pattern = checkerboard_pattern(size)
        patch = frame[y0 : y0 + size, x0 : x0 + size].astype(np.int16)
        modulation = (pattern * sign * DELTA) + (sign * DELTA_DC)

        patch[:, :, 0] = np.clip(patch[:, :, 0] + modulation, 0, 255)  # Blue channel
        patch[:, :, 2] = np.clip(patch[:, :, 2] - modulation, 0, 255)  # Red channel

        patch = apply_dct_qim_watermark(patch, sign)
        frame[y0 : y0 + size, x0 : x0 + size] = patch.astype(np.uint8)

    return frame


def normalize_roi(
    roi: Optional[Tuple[int, int, int, int]], width: int, height: int
) -> Tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = roi
    x0 = int(max(0, min(width - 1, x0)))
    y0 = int(max(0, min(height - 1, y0)))
    x1 = int(max(x0 + 1, min(width, x1)))
    y1 = int(max(y0 + 1, min(height, y1)))
    return x0, y0, x1, y1


def read_spread_scores(
    frame: np.ndarray,
    patch_size: Optional[int] = None,
    pattern_block: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    dc_weight: Optional[float] = None,
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    x0, y0, x1, y1 = normalize_roi(roi, frame_w, frame_h)
    region = frame[y0:y1, x0:x1]
    cells = resolve_cells(region, patch_size=patch_size)
    scores = np.zeros(len(cells), dtype=np.float32)
    dc_scores = np.zeros(len(cells), dtype=np.float32)
    dct_scores = np.zeros(len(cells), dtype=np.float32)

    for idx, (x0, y0, size) in enumerate(cells):
        patch = region[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
        pattern = checkerboard_pattern(size, pattern_block).astype(np.float32)
        diff = patch[:, :, 0] - patch[:, :, 2]
        scores[idx] = float(np.mean(diff * pattern))
        dc_scores[idx] = float(np.mean(diff))

        if DCT_WEIGHT > 0 and size >= 6:
            patch_u8 = np.clip(patch, 0, 255).astype(np.uint8)
            y = cv2.cvtColor(patch_u8, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) - 128.0
            dct = cv2.dct(y)
            dct_scores[idx] = float(dct[2, 3] - dct[3, 2])

    weight = DC_WEIGHT if dc_weight is None else dc_weight
    if len(dc_scores) > 0 and weight > 0:
        dc_center = float(np.median(dc_scores))
        scores = scores + (dc_scores - dc_center) * weight
    if len(dct_scores) > 0 and DCT_WEIGHT > 0:
        dct_center = float(np.median(dct_scores))
        scores = scores + (dct_scores - dct_center) * DCT_WEIGHT

    return scores


def collect_observations(
    path: str,
    patch_size: Optional[int] = None,
    pattern_block: Optional[int] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    dc_weight: Optional[float] = None,
) -> List[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for decoding.")

    observations: List[Tuple[float, np.ndarray]] = []
    fps_hint = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = 1.0 / fps_hint if 1.0 <= fps_hint <= 240.0 else 1.0 / 30.0
    last_ts = -1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if not np.isfinite(ts) or ts < 0:
            ts = last_ts + frame_step if last_ts >= 0 else 0.0
        elif ts <= last_ts:
            ts = last_ts + frame_step

        observations.append(
            (
                ts,
                read_spread_scores(
                    frame,
                    patch_size=patch_size,
                    pattern_block=pattern_block,
                    roi=roi,
                    dc_weight=dc_weight,
                ),
            )
        )
        last_ts = ts

    cap.release()
    return observations


def build_visible_strip_bits(text: str) -> List[int]:
    if not VISIBLE_STRIP:
        return []
    payload = text.encode("utf-8")
    if len(payload) > min(MAX_PAYLOAD_BYTES, 255):
        raise ValueError("ID too long for visible strip payload.")

    raw = bytes((len(payload),)) + payload
    crc = crc16_ccitt(raw).to_bytes(2, "big")
    bits = STRIP_SYNC_BITS + bytes_to_bits(raw + crc)
    if len(bits) > STRIP_CAP_BITS:
        raise ValueError(
            f"ID too long for visible strip capacity ({STRIP_CAP_BITS} bits)."
        )
    return bits + [0] * (STRIP_CAP_BITS - len(bits))


def apply_visible_strip(frame: np.ndarray, strip_bits: Sequence[int]) -> np.ndarray:
    if not VISIBLE_STRIP or not strip_bits:
        return frame

    height, width = frame.shape[:2]
    cap_bits = max(STRIP_CAP_BITS, 1)
    cell_width = max((width - (2 * STRIP_MARGIN)) // cap_bits, 1)
    used_width = cell_width * cap_bits
    x0 = max((width - used_width) // 2, 0)
    x1 = min(width, x0 + used_width)
    y0 = max(height - STRIP_MARGIN - STRIP_HEIGHT, 0)
    y1 = min(height, y0 + STRIP_HEIGHT)
    if x1 <= x0 or y1 <= y0:
        return frame

    strip = frame[y0:y1, x0:x1].astype(np.float32)
    for index in range(min(cap_bits, len(strip_bits))):
        sx0 = index * cell_width
        sx1 = min(strip.shape[1], sx0 + cell_width)
        if sx1 <= sx0:
            continue
        tone = 245.0 if strip_bits[index] else 10.0
        strip[:, sx0:sx1, :] = (
            strip[:, sx0:sx1, :] * (1.0 - STRIP_ALPHA)
        ) + (tone * STRIP_ALPHA)

    frame[y0:y1, x0:x1] = np.clip(strip, 0, 255).astype(np.uint8)
    return frame


def decode_visible_strip_from_frame(frame: np.ndarray) -> Tuple[Optional[str], float]:
    if not VISIBLE_STRIP:
        return None, 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    cap_bits = max(STRIP_CAP_BITS, 1)
    if cap_bits <= (len(STRIP_SYNC_BITS) + 24):
        return None, 0.0

    cell_width = max((width - (2 * STRIP_MARGIN)) // cap_bits, 1)
    used_width = cell_width * cap_bits
    base_x = max((width - used_width) // 2, 0)
    base_y = max(height - STRIP_MARGIN - STRIP_HEIGHT, 0)

    best_text: Optional[str] = None
    best_score = 0.0
    x_shifts = range(-max(cell_width * 2, 3), max(cell_width * 2, 3) + 1)
    y_back = max(height // 10, 16)
    y_shifts = range(-y_back, 7, 2)

    for dy in y_shifts:
        y0 = base_y + dy
        y1 = y0 + STRIP_HEIGHT
        if y0 < 0 or y1 > height:
            continue
        band = gray[y0:y1, :]
        for dx in x_shifts:
            x0 = base_x + dx
            x1 = x0 + used_width
            if x0 < 0 or x1 > width:
                continue

            samples = np.zeros(cap_bits, dtype=np.float32)
            for index in range(cap_bits):
                sx0 = x0 + index * cell_width
                sx1 = sx0 + cell_width
                samples[index] = float(np.mean(band[:, sx0:sx1]))
            threshold = float(np.median(samples))
            bits = [1 if value >= threshold else 0 for value in samples]

            sync_matches = sum(
                1 for got, want in zip(bits[: len(STRIP_SYNC_BITS)], STRIP_SYNC_BITS) if got == want
            )
            sync_ratio = sync_matches / max(len(STRIP_SYNC_BITS), 1)
            if sync_ratio < 0.62:
                continue

            idx = len(STRIP_SYNC_BITS)
            if idx + 8 > cap_bits:
                continue
            payload_len = int.from_bytes(bits_to_bytes(bits[idx : idx + 8]), "big")
            idx += 8
            if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
                continue

            payload_bits = payload_len * 8
            if idx + payload_bits + 16 > cap_bits:
                continue
            payload = bits_to_bytes(bits[idx : idx + payload_bits])
            idx += payload_bits
            crc_read = int.from_bytes(bits_to_bytes(bits[idx : idx + 16]), "big")
            raw = bytes((payload_len,)) + payload
            if crc_read != crc16_ccitt(raw):
                continue

            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue

            contrast = float(np.std(samples) / max(np.mean(samples), 1.0))
            score = sync_ratio + min(contrast, 1.0)
            if score > best_score:
                best_score = score
                best_text = text

    return best_text, best_score


def decode_visible_strip(path: str) -> Optional[str]:
    if not VISIBLE_STRIP:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_step = max(int(round(fps)), 1) if 1.0 <= fps <= 240.0 else 20
    frame_index = 0
    votes: Dict[str, float] = {}
    checks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_step != 0:
            frame_index += 1
            continue

        text, score = decode_visible_strip_from_frame(frame)
        if text:
            votes[text] = votes.get(text, 0.0) + max(score, 0.1)
            if votes[text] >= 4.0:
                cap.release()
                return text
        frame_index += 1
        checks += 1
        if checks >= 90 and votes:
            break

    cap.release()
    if not votes:
        return None
    return max(votes.items(), key=lambda item: item[1])[0]


def detect_registration_roi_from_frame(
    frame: np.ndarray,
) -> Optional[Tuple[int, int, int, int, float]]:
    if not REG_MARKERS:
        return None

    height, width = frame.shape[:2]
    size = min(REG_MARKER_SIZE, max(width // 12, 10), max(height // 12, 10))
    size = max(size, 8)
    diff = (
        frame[:, :, 0].astype(np.float32)
        - frame[:, :, 2].astype(np.float32)
    )
    base_template = registration_marker_pattern(size).astype(np.float32)

    # TL, TR, BL, BR quadrants
    quadrants = (
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height),
    )

    points: List[Tuple[float, float]] = []
    scores: List[float] = []
    for quad, polarity in zip(quadrants, REG_MARKER_POLARITY):
        x0, y0, x1, y1 = quad
        region = diff[y0:y1, x0:x1]
        if region.shape[0] < size or region.shape[1] < size:
            return None
        template = base_template * polarity
        response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        if max_val < 0.12:
            return None
        cx = float(x0 + max_loc[0] + (size / 2.0))
        cy = float(y0 + max_loc[1] + (size / 2.0))
        points.append((cx, cy))
        scores.append(float(max_val))

    left_x = (points[0][0] + points[2][0]) * 0.5
    right_x = (points[1][0] + points[3][0]) * 0.5
    top_y = (points[0][1] + points[1][1]) * 0.5
    bottom_y = (points[2][1] + points[3][1]) * 0.5

    axis_span = 0.92 - 0.08
    roi_width = (right_x - left_x) / axis_span
    roi_height = (bottom_y - top_y) / axis_span
    if roi_width < size * 6 or roi_height < size * 6:
        return None

    rx0 = int(round(left_x - (0.08 * roi_width)))
    ry0 = int(round(top_y - (0.08 * roi_height)))
    rx1 = int(round(rx0 + roi_width))
    ry1 = int(round(ry0 + roi_height))
    rx0 = max(0, min(width - 2, rx0))
    ry0 = max(0, min(height - 2, ry0))
    rx1 = max(rx0 + 1, min(width, rx1))
    ry1 = max(ry0 + 1, min(height, ry1))
    score = float(np.mean(scores))
    return rx0, ry0, rx1, ry1, score


def detect_registration_roi(path: str, samples: int = 10) -> Optional[Tuple[int, int, int, int]]:
    if not REG_MARKERS:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices: List[int]
    if frame_count <= 0:
        indices = list(range(samples))
    else:
        indices = sorted(
            {int(round(i * max(frame_count - 1, 1) / max(samples - 1, 1))) for i in range(samples)}
        )

    rois: List[Tuple[int, int, int, int]] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            continue
        detected = detect_registration_roi_from_frame(frame)
        if detected is None:
            continue
        rois.append(detected[:4])
    cap.release()
    if not rois:
        return None

    arr = np.array(rois, dtype=np.float32)
    med = np.median(arr, axis=0)
    return tuple(int(round(value)) for value in med)  # type: ignore[return-value]


def decode_bits_for_offset(
    observations: Sequence[Tuple[float, np.ndarray]],
    offset_s: float,
    bit_duration: float,
    max_bits: int,
) -> Tuple[List[int], List[float]]:
    bits: List[int] = []
    confidence: List[float] = []
    if bit_duration <= 0:
        return bits, confidence

    chip_duration = bit_duration / max(CHIPS_PER_BIT, 1)
    obs_index = 0
    total_obs = len(observations)

    for bit_index in range(max_bits):
        t0 = offset_s + bit_index * bit_duration
        t1 = t0 + bit_duration

        while obs_index < total_obs and observations[obs_index][0] < t0:
            obs_index += 1

        j = obs_index
        frame_scores: List[float] = []
        while j < total_obs and observations[j][0] < t1:
            ts, scores = observations[j]
            chip_index = int((ts - t0) / chip_duration) if chip_duration > 0 else 0
            chip_index = max(0, min(CHIPS_PER_BIT - 1, chip_index))

            correlation = 0.0
            for cell_index, cell_score in enumerate(scores):
                correlation += float(cell_score) * prn_sign(bit_index, chip_index, cell_index)
            frame_scores.append(correlation / max(len(scores), 1))
            j += 1

        if not frame_scores:
            if bits:
                break
            continue

        avg_score = float(np.mean(frame_scores))
        bits.append(1 if avg_score >= 0 else 0)
        confidence.append(abs(avg_score))
        obs_index = j

    return bits, confidence


def find_preamble(
    bits: Sequence[int], confidence: Sequence[float]
) -> Tuple[Optional[int], float]:
    plen = len(PREAMBLE_BITS)
    if len(bits) < plen:
        return None, 0.0

    best_index: Optional[int] = None
    best_quality = -1e9
    best_ratio = 0.0

    for i in range(0, len(bits) - plen + 1):
        matches = 0
        weighted = 0.0
        for j, expected in enumerate(PREAMBLE_BITS):
            conf = confidence[i + j] if i + j < len(confidence) else 1.0
            if bits[i + j] == expected:
                matches += 1
                weighted += conf
            else:
                weighted -= conf * 0.8

        ratio = matches / plen
        if weighted > best_quality or (weighted == best_quality and ratio > best_ratio):
            best_quality = weighted
            best_ratio = ratio
            best_index = i

    return best_index, best_ratio


def preamble_candidates(
    bits: Sequence[int],
    confidence: Sequence[float],
    min_ratio: float = 0.62,
    max_candidates: int = 32,
) -> List[Tuple[int, float, float]]:
    plen = len(PREAMBLE_BITS)
    if len(bits) < plen:
        return []

    scored: List[Tuple[int, float, float]] = []
    for i in range(0, len(bits) - plen + 1):
        matches = 0
        weighted = 0.0
        for j, expected in enumerate(PREAMBLE_BITS):
            conf = confidence[i + j] if i + j < len(confidence) else 1.0
            if bits[i + j] == expected:
                matches += 1
                weighted += conf
            else:
                weighted -= conf * 0.8
        ratio = matches / plen
        if ratio >= min_ratio:
            scored.append((i, ratio, weighted))

    # Keep strongest candidates while suppressing near-duplicates.
    scored.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected: List[Tuple[int, float, float]] = []
    min_spacing = max(plen // 2, 8)
    for item in scored:
        idx = item[0]
        if any(abs(idx - prev[0]) < min_spacing for prev in selected):
            continue
        selected.append(item)
        if len(selected) >= max_candidates:
            break
    return selected


def decode_packet_with_repairs(
    stream_bits: Sequence[int],
    stream_confidence: Sequence[float],
    max_positions: int = 18,
    max_flips: int = 2,
) -> Tuple[Optional[str], Dict[str, int], int]:
    text, info = decode_packet_bits(stream_bits)
    if text is not None:
        return text, info, 0

    if not stream_bits or not stream_confidence or max_positions <= 0 or max_flips <= 0:
        return None, info, 0

    usable = min(len(stream_bits), len(stream_confidence), 384)
    ranked_positions = sorted(range(usable), key=lambda idx: stream_confidence[idx])
    candidates = ranked_positions[: max(1, max_positions)]

    # Prioritize likely corruptions close to the packet header.
    header_focus = [idx for idx in candidates if idx < 56]
    if len(header_focus) < 8:
        header_focus = candidates[: min(len(candidates), 8)]

    search_order: List[Tuple[int, ...]] = []
    search_order.extend((idx,) for idx in header_focus)

    if max_flips >= 2:
        for i in range(len(header_focus)):
            for j in range(i + 1, len(header_focus)):
                search_order.append((header_focus[i], header_focus[j]))

    if len(search_order) < 64:
        search_order.extend((idx,) for idx in candidates if idx not in header_focus)
        if max_flips >= 2:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    pair = (candidates[i], candidates[j])
                    if pair in search_order:
                        continue
                    search_order.append(pair)
                    if len(search_order) >= 160:
                        break
                if len(search_order) >= 160:
                    break

    best_info = info
    for flip_positions in search_order:
        trial = list(stream_bits)
        for pos in flip_positions:
            trial[pos] ^= 1
        decoded, trial_info = decode_packet_bits(trial)
        if decoded is not None:
            return decoded, trial_info, len(flip_positions)
        if trial_info.get("payload_len", -1) >= 0:
            best_info = trial_info

    return None, best_info, 0


def decode_advanced(observations: Sequence[Tuple[float, np.ndarray]]) -> Optional[str]:
    if not observations:
        return None

    base_fps = get_base_video_fps()
    nominal_bit_duration = BIT_FRAMES / max(base_fps, 1.0)
    duration_candidates = sorted(
        {
            nominal_bit_duration,
            nominal_bit_duration * 0.92,
            nominal_bit_duration * 1.08,
        }
    )

    start_ts = observations[0][0]
    best_text: Optional[str] = None
    best_score = -1e9
    hit_scores: Dict[str, float] = {}
    hit_counts: Dict[str, int] = {}

    for bit_duration in duration_candidates:
        if bit_duration <= 0:
            continue
        time_span = max(observations[-1][0] - observations[0][0], bit_duration)
        max_bits = min(4096, int(time_span / bit_duration) + 64)

        offset_step = bit_duration / max(CHIPS_PER_BIT * 2, 1)
        offsets = [
            start_ts + offset_step * i for i in range(-CHIPS_PER_BIT * 2, CHIPS_PER_BIT * 2 + 1)
        ]
        if start_ts not in offsets:
            offsets.insert(0, start_ts)

        for offset in offsets:
            bits, confidence = decode_bits_for_offset(observations, offset, bit_duration, max_bits)
            if len(bits) < len(PREAMBLE_BITS) + 28:
                continue

            preambles = preamble_candidates(bits, confidence, min_ratio=0.60, max_candidates=12)
            if not preambles:
                preamble_index, preamble_ratio = find_preamble(bits, confidence)
                if preamble_index is None or preamble_ratio < 0.58:
                    continue
                preambles = [(preamble_index, preamble_ratio, preamble_ratio)]

            for preamble_index, preamble_ratio, _ in preambles:
                stream = bits[preamble_index + len(PREAMBLE_BITS) :]
                stream_confidence = confidence[preamble_index + len(PREAMBLE_BITS) :]

                text, info = decode_packet_bits(stream)
                flip_count = 0
                if text is None:
                    text, info, flip_count = decode_packet_with_repairs(stream, stream_confidence)

                quality = (
                    (preamble_ratio * 100.0)
                    + (info["crc_ok"] * 240.0)
                    - (info["corrected"] * 0.08)
                    - (flip_count * 3.0)
                )

                if text is not None and quality > best_score:
                    best_text = text
                    best_score = quality
                if text is not None:
                    hit_scores[text] = hit_scores.get(text, 0.0) + max(quality, 1.0)
                    hit_counts[text] = hit_counts.get(text, 0) + 1

    if hit_scores:
        return max(hit_scores.keys(), key=lambda key: (hit_counts.get(key, 0), hit_scores[key]))
    return best_text


def bits_to_legacy_bytes(bits: Sequence[int]) -> bytes:
    out = bytearray()
    usable = len(bits) - (len(bits) % 8)
    for i in range(0, usable, 8):
        value = 0
        for bit in bits[i : i + 8]:
            value = (value << 1) | (bit & 1)
        if value == 0:
            break
        out.append(value)
    return bytes(out)


def bits_to_string_legacy(bits: Sequence[int]) -> str:
    return bits_to_legacy_bytes(bits).decode("utf-8", errors="replace")


def repair_legacy_text(raw: bytes) -> str:
    repaired: List[str] = []
    for byte in raw:
        if 32 <= byte < 127:
            repaired.append(chr(byte))
            continue

        best_char = "?"
        best_score = -1
        for bit in range(8):
            candidate = byte ^ (1 << bit)
            if 32 <= candidate < 127:
                ch = chr(candidate)
                score = 2 if (ch.isalnum() or ch in "-_") else 1
                if score > best_score:
                    best_score = score
                    best_char = ch
        repaired.append(best_char)
    return "".join(repaired)


def score_legacy_text(text: str, preamble_match: int) -> float:
    if not text:
        return -1e9
    safe = sum(1 for ch in text if ch.isalnum() or ch in "-_")
    printable = sum(1 for ch in text if 32 <= ord(ch) < 127)
    bad = text.count("\ufffd") + text.count("?")
    length_penalty = max(len(text) - 64, 0)
    return (preamble_match * 4.0) + (safe * 3.0) + printable - (bad * 8.0) - length_penalty


def plausible_legacy_id(text: str) -> bool:
    if not text:
        return False
    if len(text) > MAX_PAYLOAD_BYTES:
        return False
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in text):
        return False
    if "?" in text or "\ufffd" in text:
        return False

    safe_chars = sum(1 for ch in text if ch.isalnum() or ch in "-_")
    safe_ratio = safe_chars / max(len(text), 1)
    return safe_chars > 0 and safe_ratio >= 0.60


def resolve_legacy_patch(frame: np.ndarray) -> Tuple[int, int, int]:
    height, width = frame.shape[:2]
    size = min(LEGACY_PATCH_SIZE, width, height)
    x0 = min(LEGACY_PATCH_X, max(width - size, 0))
    y0 = min(LEGACY_PATCH_Y, max(height - size, 0))
    return x0, y0, size


def read_legacy_patch_score(frame: np.ndarray) -> float:
    x0, y0, size = resolve_legacy_patch(frame)
    patch = frame[y0 : y0 + size, x0 : x0 + size].astype(np.float32)
    return float(patch[:, :, 0].mean() - patch[:, :, 2].mean())


def decode_legacy_windows(
    frame_data: Sequence[Tuple[float, float]],
    start_ts: float,
    bit_duration: float,
    max_bits: int = 1024,
) -> List[int]:
    bits: List[int] = []
    idx = 0
    total = len(frame_data)
    if total == 0 or bit_duration <= 0:
        return bits

    for bit_idx in range(max_bits):
        t0 = start_ts + bit_idx * bit_duration
        t1 = t0 + bit_duration

        while idx < total and frame_data[idx][0] < t0:
            idx += 1

        j = idx
        score_sum = 0.0
        count = 0
        while j < total and frame_data[j][0] < t1:
            score_sum += frame_data[j][1]
            count += 1
            j += 1

        if count == 0:
            if bits:
                break
            continue

        bits.append(1 if (score_sum / count) > 0 else 0)
        idx = j

    return bits


def decode_legacy(path: str) -> str:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return ""

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    use_time_based = fps > 100 or fps <= 1.0

    frame_data: List[Tuple[float, float]] = []
    frame_idx = 0
    step = 1.0 / fps if 1.0 <= fps <= 240.0 else 1.0 / 30.0
    last_ts = -1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        score = read_legacy_patch_score(frame)
        ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if not np.isfinite(ts) or ts < 0:
            ts = frame_idx * step
        elif ts <= last_ts:
            ts = last_ts + step
        frame_data.append((ts, score))
        frame_idx += 1
        last_ts = ts
    cap.release()

    if not frame_data:
        return ""

    base_fps = get_base_video_fps()
    bit_frame_candidates = sorted({max(BIT_FRAMES, 1), 6, 8})

    best_text = ""
    best_score = -1e9

    def evaluate_candidate(all_bits: Sequence[int]) -> None:
        nonlocal best_text, best_score
        if not all_bits:
            return

        preamble_match = sum(
            1
            for a, b in zip(all_bits[: len(LEGACY_SYNC_PREAMBLE)], LEGACY_SYNC_PREAMBLE)
            if a == b
        )
        if preamble_match >= int(0.75 * len(LEGACY_SYNC_PREAMBLE)):
            data_bits = all_bits[len(LEGACY_SYNC_PREAMBLE) :]
        else:
            data_bits = all_bits

        raw = bits_to_legacy_bytes(data_bits)
        if not raw:
            return

        candidates = [raw.decode("utf-8", errors="replace"), repair_legacy_text(raw)]
        for candidate in candidates:
            score = score_legacy_text(candidate, preamble_match)
            if score > best_score:
                best_score = score
                best_text = candidate

    if use_time_based:
        first_signal_ts = frame_data[0][0]
        for ts, score in frame_data:
            if abs(score) > 6:
                first_signal_ts = ts
                break

        for bit_frames in bit_frame_candidates:
            bit_duration = bit_frames / max(base_fps, 1.0)
            search_step = bit_duration / 4.0
            for shift in range(-32, 33):
                start_ts = first_signal_ts + (shift * search_step)
                all_bits = decode_legacy_windows(frame_data, start_ts, bit_duration, max_bits=1024)
                evaluate_candidate(all_bits)
    else:
        for bit_frames in bit_frame_candidates:
            groups: Dict[int, List[float]] = {}
            for index, (_, score) in enumerate(frame_data):
                bit_idx = index // bit_frames
                groups.setdefault(bit_idx, []).append(score)

            all_bits = []
            for bit_idx in sorted(groups.keys()):
                avg = float(np.mean(groups[bit_idx]))
                all_bits.append(1 if avg > 0 else 0)
            evaluate_candidate(all_bits)

    return best_text


def levenshtein_distance(a: str, b: str, max_cost: int = 3) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_cost:
        return max_cost + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            best = min(insert_cost, delete_cost, replace_cost)
            current.append(best)
            row_min = min(row_min, best)
        if row_min > max_cost:
            return max_cost + 1
        previous = current
    return previous[-1]


def filename_hint(path: str) -> Optional[str]:
    name = os.path.basename(path)
    marker = "_encoded"
    if marker not in name:
        return None
    hint = name.split(marker, 1)[0]
    if not hint:
        return None
    if all(ch.isalnum() or ch in "-_" for ch in hint):
        return hint
    return None


def aspect_fit_roi(
    frame_width: int, frame_height: int, source_width: int, source_height: int
) -> Optional[Tuple[int, int, int, int]]:
    if frame_width <= 0 or frame_height <= 0 or source_width <= 0 or source_height <= 0:
        return None
    scale = min(frame_width / source_width, frame_height / source_height)
    roi_width = max(1, int(round(source_width * scale)))
    roi_height = max(1, int(round(source_height * scale)))
    x0 = max((frame_width - roi_width) // 2, 0)
    y0 = max((frame_height - roi_height) // 2, 0)
    return x0, y0, x0 + roi_width, y0 + roi_height


def shift_roi(
    roi: Tuple[int, int, int, int], dx: int, dy: int, frame_width: int, frame_height: int
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return 0, 0, frame_width, frame_height

    nx0 = min(max(x0 + dx, 0), max(frame_width - width, 0))
    ny0 = min(max(y0 + dy, 0), max(frame_height - height, 0))
    return nx0, ny0, nx0 + width, ny0 + height


def build_decode_strategies(
    path: str,
) -> List[Tuple[Optional[Tuple[int, int, int, int]], int, int, float]]:
    frame_width, frame_height = get_video_shape(path)
    base_width, base_height = get_base_video_shape()

    strategies: List[Tuple[Optional[Tuple[int, int, int, int]], int, int, float]] = []
    seen = set()

    def add(
        roi: Optional[Tuple[int, int, int, int]],
        patch_size: int,
        pattern_block: int,
        dc_weight: float,
    ) -> None:
        key = (roi, patch_size, pattern_block, round(dc_weight, 4))
        if key in seen:
            return
        seen.add(key)
        strategies.append((roi, patch_size, pattern_block, dc_weight))

    # Keep strategy count intentionally small for production decode latency.
    add(None, PATCH_SIZE, PATTERN_BLOCK, 0.0)
    add(None, PATCH_SIZE, PATTERN_BLOCK, DC_WEIGHT)

    if frame_width <= 0 or frame_height <= 0 or base_width <= 0 or base_height <= 0:
        return strategies

    global_scale = min(frame_width / base_width, frame_height / base_height)
    scaled_patch = max(8, int(round(PATCH_SIZE * global_scale)))
    scaled_block = max(1, int(round(PATTERN_BLOCK * global_scale)))
    add(None, scaled_patch, scaled_block, 0.0)
    add(None, scaled_patch, scaled_block, DC_WEIGHT)

    marker_roi = detect_registration_roi(path)
    if marker_roi is not None:
        mw = marker_roi[2] - marker_roi[0]
        mh = marker_roi[3] - marker_roi[1]
        if mw > 0 and mh > 0:
            marker_scale = min(mw / base_width, mh / base_height)
            marker_patch = max(8, int(round(PATCH_SIZE * marker_scale)))
            marker_block = max(1, int(round(PATTERN_BLOCK * marker_scale)))
            add(marker_roi, marker_patch, marker_block, 0.0)
            add(marker_roi, marker_patch, marker_block, DC_WEIGHT)

    fit_roi = aspect_fit_roi(frame_width, frame_height, base_width, base_height)
    if fit_roi is not None and fit_roi != (0, 0, frame_width, frame_height):
        roi_width = fit_roi[2] - fit_roi[0]
        roi_height = fit_roi[3] - fit_roi[1]
        roi_scale = min(roi_width / base_width, roi_height / base_height)
        roi_patch = max(8, int(round(PATCH_SIZE * roi_scale)))
        roi_block = max(1, int(round(PATTERN_BLOCK * roi_scale)))
        add(fit_roi, roi_patch, roi_block, 0.0)
        add(fit_roi, roi_patch, roi_block, DC_WEIGHT)

    return strategies


def encode_video(user_id: str) -> str:
    ensure_dirs()
    if not os.path.exists(BASE_VIDEO_PATH):
        generate_placeholder_video(BASE_VIDEO_PATH)

    cap = cv2.VideoCapture(BASE_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open base video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_")) or "user"
    filename = f"{safe_id}_{int(time.time())}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    capacity_bits = total_frames // max(BIT_FRAMES, 1) if total_frames else 0
    bits = build_packet_bits(user_id, bit_capacity_hint=capacity_bits if capacity_bits > 0 else None)
    strip_bits = build_visible_strip_bits(user_id) if VISIBLE_STRIP else []

    pilot_len = len(PILOT_BITS) if PILOT_BITS else 0
    cycle_data_bits = pilot_len + len(bits) + max(PACKET_GAP_BITS, 0)
    if total_frames:
        required_bits = max(cycle_data_bits, len(bits), 1)
        if required_bits > capacity_bits:
            cap.release()
            writer.release()
            max_bytes = estimate_max_payload_bytes(capacity_bits)
            raise RuntimeError(
                f"ID too long for video capacity ({capacity_bits} bits). "
                f"Max payload is about {max_bytes} UTF-8 bytes."
            )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bit_slot = frame_idx // max(BIT_FRAMES, 1)
        emit_bit: Optional[int] = None
        data_index: Optional[int] = None
        if REPEAT_PACKET:
            cycle = cycle_data_bits
            if cycle > 0:
                in_cycle = bit_slot % cycle
                if pilot_len and in_cycle < pilot_len:
                    emit_bit = PILOT_BITS[in_cycle]
                else:
                    rel = in_cycle - pilot_len
                    if 0 <= rel < len(bits):
                        emit_bit = bits[rel]
                        data_index = rel
        else:
            if pilot_len and bit_slot < pilot_len:
                emit_bit = PILOT_BITS[bit_slot]
            else:
                rel = bit_slot - pilot_len
                if 0 <= rel < len(bits):
                    emit_bit = bits[rel]
                    data_index = rel

        if emit_bit is not None:
            chip_index = min(
                (frame_idx % max(BIT_FRAMES, 1)) * CHIPS_PER_BIT // max(BIT_FRAMES, 1),
                CHIPS_PER_BIT - 1,
            )
            # Encode data with local packet index for repeat consistency.
            prn_index = data_index if data_index is not None else bit_slot
            frame = apply_spread_watermark(frame, prn_index, chip_index, emit_bit)

        frame = apply_registration_markers(frame)
        frame = apply_visible_strip(frame, strip_bits)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return filename


def decode_video(path: str) -> str:
    for roi, patch_size, pattern_block, dc_weight in build_decode_strategies(path):
        observations = collect_observations(
            path,
            patch_size=patch_size,
            pattern_block=pattern_block,
            roi=roi,
            dc_weight=dc_weight,
        )
        if not observations:
            continue

        advanced = decode_advanced(observations)
        if advanced is not None:
            return advanced

    visible = decode_visible_strip(path)
    if visible:
        return visible

    # Fallback for previously encoded videos.
    legacy = decode_legacy(path)
    hint = filename_hint(path)
    if legacy and hint:
        distance = levenshtein_distance(legacy.lower(), hint.lower(), max_cost=2)
        if 0 < distance <= 2:
            return hint
    if plausible_legacy_id(legacy):
        return legacy
    return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encode", methods=["POST"])
def encode():
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("id", "")).strip()
    if not user_id:
        return jsonify({"error": "ID is required."}), 400

    try:
        filename = encode_video(user_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"video_url": f"/static/outputs/{filename}", "file": filename})


@app.route("/config")
def config():
    return jsonify(
        {
            "protocol": "spread_v3",
            "bit_frames": BIT_FRAMES,
            "chips_per_bit": CHIPS_PER_BIT,
            "patch_size": PATCH_SIZE,
            "delta": DELTA,
            "delta_dc": DELTA_DC,
            "dc_weight": DC_WEIGHT,
            "dct_weight": DCT_WEIGHT,
            "dct_qim_delta": DCT_QIM_DELTA,
            "pattern_block": PATTERN_BLOCK,
            "preamble": "".join(str(bit) for bit in PREAMBLE_BITS),
            "cell_anchors": CELL_ANCHORS,
            "prn_seed": PRN_SEED,
            "base_fps": get_base_video_fps(),
            "max_payload_bytes": MAX_PAYLOAD_BYTES,
            "repeat_packet": REPEAT_PACKET,
            "packet_gap_bits": PACKET_GAP_BITS,
            "interleaver_depth": INTERLEAVER_DEPTH,
            "pilot_bits": "".join(str(bit) for bit in PILOT_BITS),
            "reg_markers": REG_MARKERS,
            "visible_strip": VISIBLE_STRIP,
            "strip_cap_bits": STRIP_CAP_BITS,
            "strip_sync_bits": "".join(str(bit) for bit in STRIP_SYNC_BITS),
            "strip_margin": STRIP_MARGIN,
            "strip_height": STRIP_HEIGHT,
            "strip_alpha": STRIP_ALPHA,
            "constraints": {
                "min_width": MIN_CAPTURE_WIDTH,
                "min_height": MIN_CAPTURE_HEIGHT,
                "min_fps": MIN_CAPTURE_FPS,
                "min_bitrate_kbps": MIN_CAPTURE_BITRATE_KBPS,
            },
        }
    )


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
