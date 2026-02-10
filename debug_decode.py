"""Simple decoder debug helper."""

import sys
from app import decode_video


def safe_print(text: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("unicode_escape").decode("ascii"))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python debug_decode.py <video_path>")
        return 1

    path = sys.argv[1]
    try:
        value = decode_video(path)
    except Exception as exc:
        print(f"Decode error: {exc}")
        return 1

    safe_print(f"Decoded length: {len(value)}")
    safe_print(f"Decoded repr: {ascii(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
