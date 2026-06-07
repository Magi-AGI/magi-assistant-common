"""
Offline session-transcript recovery for magi-assistant voice recordings.

Re-transcribes a Discord per-user OGG/Opus track with WhisperX (large-v3) when the
live cloud STT failed, biases spelling with a campaign phrase list, diarizes with
pyannote, and — crucially — remaps WhisperX file-relative timestamps back to
wall-clock using the audio_speech_bursts table.

Why the remap matters: the recorder writes ONE Opus frame per received packet and
does NOT pad silence (recorder.ts: frameCount++ only on data). So the OGG is
gap-compressed — file-time != wall-clock. Each speech burst gives an anchor pair:
    file_seconds = frame_offset * 0.02   (20 ms/frame)  <->  burst wall-clock
Within a burst the rate is 1:1 (20 ms frame == 20 ms real); between bursts the file
is contiguous but wall-clock jumps. We interpolate each segment's file-time onto
wall-clock across those anchors.

Usage:
    python transcribe_session.py \
        --ogg gm_track.ogg \
        --bursts bursts.json \
        --out <prefix> \
        --prompt-file prompts/dominos-fall-group2.txt \
        --min-speakers 2 --max-speakers 6

bursts.json: array of {start_frame_offset, end_frame_offset, burst_start, burst_end}
(ISO-8601 UTC), e.g. sqlite3 -json "SELECT start_frame_offset,end_frame_offset,
burst_start,burst_end FROM audio_speech_bursts WHERE track_id=? ORDER BY start_frame_offset".

Env: HF_TOKEN (pyannote diarization model is license-gated).

Outputs:
    <prefix>.json           full WhisperX result with per-segment wallclock fields
    <prefix>.wallclock.txt  merged readable transcript, "HH:MM:SS  SPEAKER  text" (UTC)
    <prefix>.raw.txt        one line per segment, file-time + wallclock + speaker
"""
import argparse
import bisect
import gc
import json
import os
from datetime import datetime, timezone

import torch
import whisperx

FRAME_MS = 20  # Discord Opus frame duration; recorder warns if a packet differs.


def parse_iso(s: str) -> float:
    """ISO-8601 (with trailing Z or offset) -> POSIX epoch seconds (float)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class IdentityClock:
    """For continuous recordings (no silence suppression): file-seconds map straight
    to wall-clock by adding a fixed base epoch (0 => timestamps are offsets from start)."""

    def __init__(self, base_epoch: float = 0.0):
        self.base = base_epoch

    def to_wall(self, ft: float) -> float:
        return self.base + ft


class BurstClock:
    """Piecewise map from WhisperX file-seconds to wall-clock epoch seconds."""

    def __init__(self, bursts):
        anchors = []  # (file_start, file_end, wall_start_epoch)
        for b in bursts:
            sf = b.get("start_frame_offset")
            ef = b.get("end_frame_offset")
            bs = b.get("burst_start")
            if sf is None or bs is None:
                continue
            # An open burst (still speaking at stop) may lack end frame/time.
            if ef is None:
                ef = sf
            file_start = sf * FRAME_MS / 1000.0
            file_end = ef * FRAME_MS / 1000.0
            anchors.append((file_start, file_end, parse_iso(bs)))
        anchors.sort(key=lambda a: a[0])
        if not anchors:
            raise SystemExit("No usable bursts — cannot remap to wall-clock.")
        self.anchors = anchors
        self.file_starts = [a[0] for a in anchors]

    def to_wall(self, ft: float) -> float:
        """File-seconds -> wall-clock epoch seconds."""
        i = bisect.bisect_right(self.file_starts, ft) - 1
        if i < 0:
            # Before the first burst: clamp to its wall-clock start.
            return self.anchors[0][2]
        file_start, file_end, wall_start = self.anchors[i]
        if ft <= file_end:
            return wall_start + (ft - file_start)  # inside burst: 1:1
        # In a micro-gap after burst i: snap to whichever boundary is nearer.
        if i + 1 < len(self.anchors):
            nxt = self.anchors[i + 1]
            return nxt[2] if (nxt[0] - ft) < (ft - file_end) else (wall_start + (file_end - file_start))
        return wall_start + (file_end - file_start)  # past last burst: clamp to its end


def hms(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ogg", required=True)
    ap.add_argument("--bursts", help="audio_speech_bursts JSON for gap-compressed "
                    "Discord tracks; omit for continuous recordings.")
    ap.add_argument("--session-start", help="ISO-8601 session start; only used when "
                    "--bursts is omitted, to anchor timestamps to wall-clock.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-file")
    ap.add_argument("--speaker", help="Known speaker label for a single-speaker track "
                    "(e.g. one per-user Discord track). When set, diarization is skipped "
                    "and every segment is attributed to this label.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--max-speakers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.speaker and not hf:
        raise SystemExit("HF_TOKEN not set — pyannote diarization is license-gated. "
                         "(Or pass --speaker for a single-speaker track.)")

    prompt = None
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[i] device={device} compute_type={compute_type} torch={torch.__version__}", flush=True)

    if args.bursts:
        with open(args.bursts, encoding="utf-8") as f:
            clock = BurstClock(json.load(f))
        print(f"[i] {len(clock.anchors)} burst anchors; "
              f"file span 0..{clock.anchors[-1][1]:.0f}s", flush=True)
    else:
        base = parse_iso(args.session_start) if args.session_start else 0.0
        clock = IdentityClock(base)
        print(f"[i] no bursts — continuous-audio mode, "
              f"base={'session start' if base else 'zero offset'}", flush=True)

    audio = whisperx.load_audio(args.ogg)
    aligned_cache = args.out + ".aligned.json"

    if os.path.exists(aligned_cache):
        print(f"[1-2/4] loading cached aligned result {aligned_cache}", flush=True)
        with open(aligned_cache, encoding="utf-8") as f:
            result = json.load(f)
    else:
        print("[1/4] transcribing (large-v3)...", flush=True)
        asr_opts = {"initial_prompt": prompt} if prompt else None
        model = whisperx.load_model("large-v3", device, compute_type=compute_type,
                                    asr_options=asr_opts)
        result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
        print(f"      {len(result['segments'])} raw segments", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

        print("[2/4] aligning...", flush=True)
        am, meta = whisperx.load_align_model(language_code=args.language, device=device)
        result = whisperx.align(result["segments"], am, meta, audio, device,
                                return_char_alignments=False)
        del am; gc.collect(); torch.cuda.empty_cache()
        with open(aligned_cache, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"      cached -> {aligned_cache}", flush=True)

    if args.speaker:
        print(f"[3/4] single-speaker track — labelling all segments '{args.speaker}' "
              f"(diarization skipped)", flush=True)
        for seg in result["segments"]:
            seg["speaker"] = args.speaker
    else:
        print(f"[3/4] diarizing ({args.min_speakers}-{args.max_speakers} speakers)...", flush=True)
        from whisperx.diarize import DiarizationPipeline
        diar = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1",
                                   token=hf, device=device)
        diar_segments = diar(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        result = whisperx.assign_word_speakers(diar_segments, result)

    print("[4/4] remapping to wall-clock + writing...", flush=True)
    for seg in result["segments"]:
        seg["file_start"] = seg.get("start", 0.0)
        seg["file_end"] = seg.get("end", seg.get("start", 0.0))
        seg["wall_start"] = clock.to_wall(seg["file_start"])
        seg["wall_end"] = clock.to_wall(seg["file_end"])

    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(args.out + ".raw.txt", "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            spk = seg.get("speaker", "UNKNOWN")
            f.write(f"[file {seg['file_start']:8.2f}] [{hms(seg['wall_start'])} UTC] "
                    f"{spk}: {seg['text'].strip()}\n")

    # Merged readable transcript: fold consecutive same-speaker segments.
    blocks, cur, buf, start_wall = [], None, [], 0.0
    def flush():
        if buf:
            blocks.append((cur, start_wall, " ".join(x.strip() for x in buf).strip()))
            buf.clear()
    for seg in result["segments"]:
        spk = seg.get("speaker", "UNKNOWN")
        if spk != cur:
            flush(); cur = spk; start_wall = seg["wall_start"]
        buf.append(seg["text"])
    flush()

    with open(args.out + ".wallclock.txt", "w", encoding="utf-8") as f:
        for spk, wall, text in blocks:
            f.write(f"{hms(wall)}\t{spk}\t{text}\n")

    speakers = sorted({s.get("speaker", "?") for s in result["segments"]})
    print(f"[done] {len(result['segments'])} segments, {len(blocks)} blocks, "
          f"speakers={speakers}", flush=True)
    print(f"[done] wrote {args.out}.json / .wallclock.txt / .raw.txt", flush=True)


if __name__ == "__main__":
    main()
