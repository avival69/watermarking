import sys

from app import capture_constraints_report, decode_video


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
        print("Usage: python decoder.py <video_path>")
        return 1
    path = sys.argv[1]
    try:
        user_id = decode_video(path)
    except Exception as exc:
        print(f"Decode failed: {exc}")
        return 1
    report = capture_constraints_report(path)
    violations = report.get("violations", [])
    if not user_id:
        if violations:
            print(
                "Capture constraints warning: "
                + ", ".join(str(v) for v in violations)
                + "."
            )
        print(
            "Decode failed: no watermark detected. "
            "Record after encoding starts, or decode the server-downloaded MP4."
        )
        return 1
    if violations:
        print(
            "Capture constraints warning: "
            + ", ".join(str(v) for v in violations)
            + "."
        )
    safe_print(user_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
