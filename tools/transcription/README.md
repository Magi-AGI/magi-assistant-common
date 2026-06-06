# Offline session-transcript recovery

Re-transcribes magi-assistant voice recordings with **WhisperX (large-v3)** when the
live cloud STT failed (e.g. a GCP billing lapse), biases spelling with a per-campaign
phrase list, diarizes with **pyannote**, and remaps timestamps back to **wall-clock**.

The audio survives STT outages, so a lost transcript is always recoverable from the
saved per-track OGGs (local `…/magi-assistant-discord/data/sessions/<uuid>/` and S3
`discord-sessions/<uuid>/`).

## Why the wall-clock remap matters

The Discord recorder writes **one Opus frame per received packet and does not pad
silence** (`recorder.ts`: `frameCount++` only on data). So a per-user OGG is
*gap-compressed* — file-time ≠ wall-clock. Each speech burst in the
`audio_speech_bursts` table gives an anchor pair:

```
file_seconds = frame_offset * 0.02   (20 ms/frame)   <->   burst wall-clock
```

Within a burst the rate is 1:1; between bursts the file is contiguous but wall-clock
jumps. `transcribe_session.py` interpolates each segment's file-time across those
anchors. For **continuous** recordings (no silence suppression, e.g. a straight WAV),
omit `--bursts` and pass `--session-start` to anchor the offsets instead.

## Setup

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export HF_TOKEN=...        # pyannote diarization is license-gated
```

## Usage

Export the burst anchors for the track, then run:

```bash
sqlite3 -json bot.sqlite \
  "SELECT start_frame_offset, end_frame_offset, burst_start, burst_end \
   FROM audio_speech_bursts WHERE track_id=<ID> ORDER BY start_frame_offset" > bursts.json

python transcribe_session.py \
  --ogg gm_track.ogg --bursts bursts.json --out <prefix> \
  --prompt-file prompts/<campaign>.txt --min-speakers 2 --max-speakers 6
```

Continuous audio (no burst table):

```bash
python transcribe_session.py --ogg talk.ogg --out <prefix> \
  --prompt-file prompts/<campaign>.txt --session-start 2026-05-27T18:00:00Z
```

Outputs: `<prefix>.json` (full result with per-segment `wall_start/wall_end`),
`<prefix>.wallclock.txt` (merged `HH:MM:SS  speaker  text`), `<prefix>.raw.txt`.

## Notes

- **Phrase lists** (`prompts/*.txt`) bias Whisper toward correct proper-noun spelling.
  Use the campaign-matching file — priming with the wrong vocab mangles names.
- **Diarization is best-effort on a single shared room mic** (in-person groups): the GM
  separates cleanly but similar-aged players may collapse into fewer clusters. Labels
  are placeholders to be remapped later; the transcript is re-mergeable without
  re-transcribing.
- Card-building (session log + nested transcript-hour cards) is done separately; for
  large/verbatim transcript cards, push via the Decko `runner` + `File.read` rather
  than inline MCP `create_card`, which condenses long content.
