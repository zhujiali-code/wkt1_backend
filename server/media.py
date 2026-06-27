"""WAV and JPEG parsing helpers used by the HTTP endpoints.

The handlers only need to know whether uploaded bytes are valid and where they
were saved. Low-level RIFF/JPEG marker parsing stays here so endpoint code can
focus on request state and business flow.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class WavInfo:
    """Essential WAV metadata extracted from RIFF chunks."""

    audio_format: int
    channels: int
    sample_rate: int
    bits_per_sample: int
    data_offset: int
    data_size: int


@dataclass
class JpegInfo:
    """JPEG dimensions and encoding mode."""

    width: int | None = None
    height: int | None = None
    progressive: bool = False


def log(message: str) -> None:
    """Local timestamped log helper for upload diagnostics."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_u32(data: bytes, offset: int) -> int:
    """Read a little-endian unsigned 32-bit integer."""
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def read_u16_be(data: bytes, offset: int) -> int:
    """Read a big-endian unsigned 16-bit integer."""
    return (data[offset] << 8) | data[offset + 1]


def parse_wav(body: bytes) -> WavInfo | None:
    """Parse a PCM WAV header and tolerate extra chunks before ``data``."""
    if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return None

    pos = 12
    audio_format = channels = sample_rate = bits_per_sample = None
    data_offset = data_size = None

    while pos + 8 <= len(body):
        chunk_id = body[pos : pos + 4]
        chunk_size = read_u32(body, pos + 4)
        chunk_data = pos + 8
        chunk_end = chunk_data + chunk_size
        if chunk_end > len(body):
            return None

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return None
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from(
                "<HHIIHH", body, chunk_data
            )
        elif chunk_id == b"data":
            data_offset = chunk_data
            data_size = chunk_size
            break

        pos = chunk_end + (chunk_size & 1)

    if (
        audio_format is None
        or channels is None
        or sample_rate is None
        or bits_per_sample is None
        or data_offset is None
        or data_size is None
    ):
        return None

    return WavInfo(
        audio_format=audio_format,
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        data_offset=data_offset,
        data_size=data_size,
    )


def pcm16_stats(pcm: bytes) -> str:
    """Return compact signal statistics for 16-bit PCM diagnostics."""
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return "samples=0"

    samples = struct.unpack_from(f"<{sample_count}h", pcm[: sample_count * 2])
    min_v = min(samples)
    max_v = max(samples)
    mean = sum(samples) / sample_count
    rms = math.sqrt(sum(s * s for s in samples) / sample_count)
    peak = max(abs(min_v), abs(max_v))
    clipped = sum(1 for s in samples if s <= -32760 or s >= 32760)
    zero_cross = sum(1 for prev, cur in zip(samples, samples[1:]) if (prev < 0 <= cur) or (prev > 0 >= cur))
    zcr = zero_cross / max(sample_count - 1, 1)

    return (
        f"samples={sample_count} min={min_v} max={max_v} "
        f"mean={mean:.1f} rms={rms:.1f} peak={peak} "
        f"clipped={clipped} zcr={zcr:.3f}"
    )


def save_wav(body: bytes, save_dir: Path) -> Path:
    """Save a WAV upload with a timestamped filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"ai_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
    path.write_bytes(body)
    return path


def validate_and_log_wav(body: bytes, save_dir: Path, prefix: str) -> tuple[bool, Path | None]:
    """Validate, save, and log a WAV upload."""
    wav = parse_wav(body)
    if wav is None:
        log(f"{prefix} 无效 WAV len={len(body)}")
        return False, None

    pcm = body[wav.data_offset : wav.data_offset + wav.data_size]
    duration = 0.0
    if wav.sample_rate > 0 and wav.channels > 0 and wav.bits_per_sample > 0:
        bytes_per_sample = wav.channels * wav.bits_per_sample // 8
        if bytes_per_sample > 0:
            duration = wav.data_size / bytes_per_sample / wav.sample_rate

    save_path = save_wav(body, save_dir)
    stats = pcm16_stats(pcm) if wav.audio_format == 1 and wav.bits_per_sample == 16 else "pcm_stats=unsupported"
    log(
        f"{prefix} WAV fmt={wav.audio_format} ch={wav.channels} rate={wav.sample_rate} "
        f"bits={wav.bits_per_sample} data={wav.data_size} duration={duration:.2f}s "
        f"{stats} saved={save_path}"
    )
    return True, save_path


def build_pcm_wav(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    add_extra_chunk: bool,
) -> bytes:
    """Wrap raw PCM bytes as a WAV file."""
    if bits_per_sample % 8 != 0:
        raise ValueError("bits_per_sample 必须是 8 的倍数")

    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
    chunks = [fmt_chunk]

    if add_extra_chunk:
        # Exercises clients that incorrectly assume the data chunk starts at byte 44.
        junk_payload = b"stream-test-extra"
        chunks.append(struct.pack("<4sI", b"JUNK", len(junk_payload)) + junk_payload)
        if len(junk_payload) & 1:
            chunks.append(b"\x00")

    chunks.append(struct.pack("<4sI", b"data", len(pcm)) + pcm)
    body = b"".join(chunks)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def make_ai_reply_wav(upload_wav: bytes, repeat: int, add_extra_chunk: bool) -> bytes | None:
    """Build a loopback test reply from an uploaded PCM WAV."""
    wav = parse_wav(upload_wav)
    if wav is None or wav.audio_format != 1:
        return None
    pcm = upload_wav[wav.data_offset : wav.data_offset + wav.data_size]
    if repeat > 1:
        pcm = pcm * repeat
    return build_pcm_wav(
        pcm,
        sample_rate=wav.sample_rate,
        channels=wav.channels,
        bits_per_sample=wav.bits_per_sample,
        add_extra_chunk=add_extra_chunk,
    )


def parse_jpeg(body: bytes) -> JpegInfo | None:
    """Parse JPEG markers and return dimensions when available."""
    if len(body) < 4 or body[:2] != b"\xFF\xD8" or body[-2:] != b"\xFF\xD9":
        return None

    pos = 2
    while pos + 4 <= len(body):
        if body[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(body) and body[pos] == 0xFF:
            pos += 1
        if pos >= len(body):
            break

        marker = body[pos]
        pos += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(body):
            return None

        segment_len = read_u16_be(body, pos)
        if segment_len < 2 or pos + segment_len > len(body):
            return None

        if marker in (0xC0, 0xC1, 0xC2):
            if segment_len < 7:
                return None
            height = read_u16_be(body, pos + 3)
            width = read_u16_be(body, pos + 5)
            return JpegInfo(width=width, height=height, progressive=(marker == 0xC2))

        pos += segment_len

    return JpegInfo()


def save_jpeg(body: bytes, save_dir: Path) -> Path:
    """Save a JPEG upload with a timestamped filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"camera_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path.write_bytes(body)
    return path


def save_camera_raw(body: bytes, save_dir: Path) -> Path:
    """Save invalid camera bytes for later debugging."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"camera_upload_invalid_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bin"
    path.write_bytes(body)
    return path


def validate_and_log_jpeg(body: bytes, save_dir: Path, prefix: str) -> tuple[bool, Path | None, JpegInfo | None]:
    """Validate, save, and log a JPEG upload."""
    jpeg = parse_jpeg(body)
    if jpeg is None:
        save_path = save_camera_raw(body, save_dir)
        log(
            f"{prefix} 无效 JPEG len={len(body)} "
            f"soi={body[:2].hex()} eoi={body[-2:].hex() if len(body) >= 2 else ''} "
            f"saved_raw={save_path}"
        )
        return False, save_path, None

    save_path = save_jpeg(body, save_dir)
    size_text = f"{jpeg.width}x{jpeg.height}" if jpeg.width and jpeg.height else "unknown"
    log(f"{prefix} JPEG len={len(body)} size={size_text} progressive={int(jpeg.progressive)} saved={save_path}")
    return True, save_path, jpeg
