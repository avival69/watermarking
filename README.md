# Real-Time Video Watermarking (Production v3)

This Flask app encodes a user ID into video with a production-oriented pipeline:

- Multi-cell spread spectrum + DCT-QIM hybrid embedding
- Continuous packet repetition with pilot burst support
- Interleaved packet coding + CRC32 integrity
- Decode-side majority voting across packet hits
- Geometric registration markers for crop/shift/scale tolerance
- Visible micro-strip fallback channel for high-reliability recovery
- Decode constraint checks (resolution/FPS/bitrate hints)
- Packet format with:
  - 32-bit preamble for alignment
  - header (magic/version/flags/length)
  - CRC32 integrity check
  - Hamming(7,4) forward error correction

## Quick Start

1. Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Add a base video at `assets/base.mp4`.

If missing, the app creates a placeholder video on first run.

3. Run the server:

```bash
python app.py
```

Open `http://localhost:5000`.

The browser shows a real-time invisible watermark preview on canvas, and also creates
a reliable server-side encoded MP4 for download (preferred for decoding).

## Decode a Video

```bash
python decoder.py <video_path>
```

## Test Encode/Decode

```bash
python test_encode_decode.py
```

## Tuning (Environment Variables)

- `BIT_FRAMES` (default `8`)
- `CHIPS_PER_BIT` (default `4`)
- `PATCH_SIZE` (default `28`)
- `DELTA` (default `7`)
- `DELTA_DC` (default `3`)
- `DC_WEIGHT` (default `0.60`)
- `DCT_WEIGHT` (default `0.55`)
- `DCT_QIM_DELTA` (default `14`)
- `PATTERN_BLOCK` (default `7`)
- `MAX_PAYLOAD_BYTES` (default `128`)
- `PRN_SEED` (default `915131`)
- `REPEAT_PACKET` (default `1`, repeats packet across full video)
- `PACKET_GAP_BITS` (default `0`, optional silent gap between repeats)
- `INTERLEAVER_DEPTH` (default `8`)
- `PILOT_BITS` (default `101011001101`)
- `REG_MARKERS` (default `1`)
- `VISIBLE_STRIP` (default `1`)
- `STRIP_CAP_BITS` (default `512`)
- `MIN_CAPTURE_WIDTH` (default `960`)
- `MIN_CAPTURE_HEIGHT` (default `540`)
- `MIN_CAPTURE_FPS` (default `20`)
- `MIN_CAPTURE_BITRATE_KBPS` (default `1500`)

Example:

```bash
BIT_FRAMES=10 CHIPS_PER_BIT=5 DELTA=6 python app.py
```

## Notes

- Keep the encoding tab visible while real-time encoding runs.
- Decoder keeps backward compatibility with older videos created by the previous algorithm.
