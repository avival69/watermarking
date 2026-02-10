"""
Test script to verify advanced encode/decode functionality.
"""

import os
import sys

import cv2

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (  # noqa: E402
    OUTPUT_DIR,
    PREAMBLE_BITS,
    build_packet_bits,
    decode_packet_bits,
    decode_video,
    encode_video,
    ensure_dirs,
)


def test_packet_roundtrip() -> bool:
    samples = [
        "user123",
        "test_user",
        "hello-world",
        "ID_2026",
        "ascii_only",
    ]

    print("Packet round-trip checks")
    print("-" * 30)

    for value in samples:
        bits = build_packet_bits(value)
        if bits[: len(PREAMBLE_BITS)] != PREAMBLE_BITS:
            print(f"  FAIL preamble mismatch for '{value}'")
            return False

        decoded, info = decode_packet_bits(bits[len(PREAMBLE_BITS) :])
        if decoded != value or info.get("crc_ok") != 1:
            print(f"  FAIL packet decode for '{value}' -> {decoded!r}, info={info}")
            return False

        print(f"  PASS '{value}'")

    return True


def test_video_encode_decode() -> bool:
    test_ids = [
        "user123",
        "HelloWorld",
    ]

    print("\nVideo encode/decode checks")
    print("-" * 30)

    ensure_dirs()
    passed = 0
    failed = 0

    for test_id in test_ids:
        print(f"\nTesting ID: '{test_id}'")
        try:
            filename = encode_video(test_id)
            video_path = os.path.join(OUTPUT_DIR, filename)
            decoded_id = decode_video(video_path)
            print(f"  Decoded: {decoded_id!r}")

            if decoded_id == test_id:
                print("  PASS")
                passed += 1
            else:
                print(f"  FAIL expected {test_id!r}, got {decoded_id!r}")
                failed += 1

            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


def write_simulated_screen_recording(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    letterbox: bool,
    add_overlay: bool = False,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open output video: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if letterbox:
            scale = min(width / max(src_w, 1), height / max(src_h, 1))
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            canvas[:] = 0
            x0 = (width - new_w) // 2
            y0 = (height - new_h) // 2
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
            out = canvas
        else:
            out = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        if add_overlay:
            overlay = out.copy()
            top_h = max(18, height // 22)
            toolbar_w = max(36, width // 26)
            cv2.rectangle(overlay, (0, 0), (width, top_h), (15, 15, 15), -1)
            cv2.rectangle(overlay, (width - toolbar_w, 0), (width, height), (22, 22, 22), -1)
            out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
            for cy in range(height // 4, height - (height // 8), max(height // 10, 28)):
                cv2.circle(out, (width - (toolbar_w // 2), cy), max(toolbar_w // 8, 4), (245, 245, 245), 2)

        writer.write(out)

    cap.release()
    writer.release()


def test_screen_record_decode() -> bool:
    print("\nScreen-recording robustness checks")
    print("-" * 30)

    ensure_dirs()
    test_id = "screen1"
    encoded_path = ""
    resize_path = ""
    letterbox_path = ""
    overlay_path = ""

    try:
        filename = encode_video(test_id)
        encoded_path = os.path.join(OUTPUT_DIR, filename)
        resize_path = os.path.join(OUTPUT_DIR, f"{test_id}_resize_test.mp4")
        letterbox_path = os.path.join(OUTPUT_DIR, f"{test_id}_letterbox_test.mp4")
        overlay_path = os.path.join(OUTPUT_DIR, f"{test_id}_overlay_test.mp4")

        write_simulated_screen_recording(encoded_path, resize_path, 1280, 720, letterbox=False)
        write_simulated_screen_recording(encoded_path, letterbox_path, 1280, 800, letterbox=True)
        write_simulated_screen_recording(
            encoded_path, overlay_path, 1280, 720, letterbox=False, add_overlay=True
        )

        decoded_resize = decode_video(resize_path)
        decoded_letterbox = decode_video(letterbox_path)
        decoded_overlay = decode_video(overlay_path)
        print(f"  Resize decode: {decoded_resize!r}")
        print(f"  Letterbox decode: {decoded_letterbox!r}")
        print(f"  Overlay decode: {decoded_overlay!r}")

        if (
            decoded_resize != test_id
            or decoded_letterbox != test_id
            or decoded_overlay != test_id
        ):
            print(f"  FAIL expected {test_id!r}")
            return False

        print("  PASS")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False
    finally:
        for path in (encoded_path, resize_path, letterbox_path, overlay_path):
            if path and os.path.exists(path):
                os.remove(path)


def main() -> int:
    packet_ok = test_packet_roundtrip()
    video_ok = test_video_encode_decode()
    screen_ok = test_screen_record_decode()
    return 0 if (packet_ok and video_ok and screen_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
