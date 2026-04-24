import { SpeechClient, protos } from '@google-cloud/speech';
import type { SttEngine, SttStream, TranscriptCallback, TranscriptEvent, SttConfig } from './types.js';
import { logger } from '../logger.js';

type IStreamingRecognizeResponse = protos.google.cloud.speech.v1.IStreamingRecognizeResponse;

/**
 * Google Cloud Speech-to-Text V2 engine for per-user mode.
 * Creates one streaming recognition session per VadGate stream open/close cycle.
 */
export class GoogleCloudSttEngine implements SttEngine {
  protected client: SpeechClient;
  protected config: SttConfig;

  constructor(config: SttConfig) {
    this.config = config;

    const clientOptions: { projectId?: string; keyFilename?: string } = {};
    if (config.googleCloud.projectId) {
      clientOptions.projectId = config.googleCloud.projectId;
    }
    if (config.googleCloud.keyFile) {
      clientOptions.keyFilename = config.googleCloud.keyFile;
    }

    this.client = new SpeechClient(clientOptions);
    logger.info('Google Cloud STT engine initialized');
  }

  createStream(userId: string, streamSequence: number, onTranscript: TranscriptCallback): SttStream {
    return new GoogleSttStream(
      this.client,
      this.config,
      userId,
      streamSequence,
      onTranscript
    );
  }

  destroy(): void {
    this.client.close().catch((err: unknown) => {
      logger.warn('Error closing Google Cloud STT client:', err);
    });
  }
}

class GoogleSttStream implements SttStream {
  private recognizeStream: ReturnType<SpeechClient['streamingRecognize']> | null = null;
  private _open = true;
  /** Set on first audio write (not constructor) to avoid gRPC handshake latency drift. */
  private streamOpenedAt: Date | null = null;
  private lastResultEndMs = 0;
  /**
   * In-memory cache of the most recent interim (non-final) event we've seen per
   * (resultEndTime anchor, resultIndex) key. On stream error/end, these are flushed
   * downstream before the stream is marked closed, so VadGate's consumer (which
   * persists to transcript_segments) can salvage partial results even if the
   * stream 408s out before the server ever emits a final.
   *
   * The key is stable across interims for the same recognizer hypothesis, so
   * later (more refined) interims naturally overwrite earlier partial text.
   */
  private pendingInterims = new Map<string, TranscriptEvent>();
  onClose?: () => void;

  constructor(
    client: SpeechClient,
    private config: SttConfig,
    private userId: string,
    private streamSequence: number,
    private onTranscript: TranscriptCallback
  ) {

    const streamingConfig = {
      config: {
        encoding: 'LINEAR16' as const,
        sampleRateHertz: config.googleCloud.sampleRateHertz,
        languageCode: config.googleCloud.languageCode,
        model: config.googleCloud.model,
        enableAutomaticPunctuation: config.googleCloud.enableAutomaticPunctuation,
        // Phrase hints boost recognition of character names, place names, game terms
        ...(config.googleCloud.phraseHints.length > 0 && {
          speechContexts: [{ phrases: config.googleCloud.phraseHints }],
        }),
      },
      interimResults: true,
    };

    this.recognizeStream = client.streamingRecognize(streamingConfig);

    this.recognizeStream.on('data', (response: IStreamingRecognizeResponse) => {
      if (!response.results || response.results.length === 0) return;

      for (let resultIdx = 0; resultIdx < response.results.length; resultIdx++) {
        const result = response.results[resultIdx];
        if (!result.alternatives || result.alternatives.length === 0) continue;

        const alt = result.alternatives[0];
        const isFinal = result.isFinal === true;

        let segmentStart: string;
        let segmentEnd: string | null = null;
        if (result.resultEndTime) {
          const anchor = this.streamOpenedAt ?? new Date();
          const offsetMs = (Number(result.resultEndTime.seconds ?? 0) * 1000) +
            Math.floor(Number(result.resultEndTime.nanos ?? 0) / 1_000_000);
          const startTime = new Date(anchor.getTime() + this.lastResultEndMs);
          const endTime = new Date(anchor.getTime() + offsetMs);
          segmentStart = startTime.toISOString();
          segmentEnd = isFinal ? endTime.toISOString() : null;
          if (isFinal) {
            this.lastResultEndMs = offsetMs;
          }
        } else {
          segmentStart = new Date().toISOString();
          segmentEnd = isFinal ? segmentStart : null;
        }

        const sttResultId = result.resultEndTime
          ? `${result.resultEndTime.seconds ?? 0}_${result.resultEndTime.nanos ?? 0}`
          : null;

        const event: TranscriptEvent = {
          userId: this.userId,
          displayName: null,
          speakerLabel: null,
          segmentStart,
          segmentEnd,
          transcript: alt.transcript ?? '',
          confidence: typeof alt.confidence === 'number' ? alt.confidence : null,
          isFinal,
          sttResultId,
          streamSequence: this.streamSequence,
          sttEngine: 'google-cloud-stt',
          sttModel: config.googleCloud.model,
        };

        // Track the latest interim per recognizer hypothesis so we can flush
        // if the stream dies before the server sends a final.
        // Key includes resultIdx because multiple simultaneous hypotheses can
        // be returned in the same response.
        const interimKey = `${sttResultId ?? 'none'}_${resultIdx}`;
        if (isFinal) {
          this.pendingInterims.delete(interimKey);
        } else {
          this.pendingInterims.set(interimKey, event);
        }

        this.onTranscript(event);
      }
    });

    this.recognizeStream.on('error', (err: Error) => {
      // Code 11 = OUT_OF_RANGE is normal for stream duration limit
      if ('code' in err && (err as { code: number }).code === 11) {
        logger.debug(`Google STT stream ended (duration limit) for user ${userId}`);
      } else {
        logger.error(`Google STT stream error for user ${userId}:`, err);
      }
      this.flushPendingInterims('stream error');
      this._open = false;
      this.onClose?.();
    });

    this.recognizeStream.on('end', () => {
      this.flushPendingInterims('stream end');
      this._open = false;
      this.onClose?.();
    });
  }

  /**
   * Re-emit any held interim results as non-final TranscriptEvents before the
   * stream closes. The downstream TranscriptWriter UPSERTs on the same
   * (session_id, track_id, user_id, stream_sequence, stt_result_id) key that
   * the original emission used — so this is idempotent with any interim that
   * was already persisted, and guarantees noisy partials survive a 408 or EOS
   * that would otherwise discard them.
   *
   * Runs inside a try/catch because the downstream handler can throw and we
   * must not prevent stream closure (which would leak gRPC resources).
   */
  private flushPendingInterims(reason: string): void {
    if (this.pendingInterims.size === 0) return;
    const count = this.pendingInterims.size;
    logger.warn(
      `Google STT stream for user ${this.userId} (seq ${this.streamSequence}): ` +
      `flushing ${count} held interim result(s) on ${reason} to preserve partial transcripts`
    );
    for (const event of this.pendingInterims.values()) {
      try {
        this.onTranscript(event);
      } catch (err) {
        logger.warn(`Google STT stream for user ${this.userId}: error flushing interim:`, err);
      }
    }
    this.pendingInterims.clear();
  }

  write(pcm: Buffer): void {
    if (!this._open || !this.recognizeStream) return;
    // Anchor stream time to first audio byte, not constructor (avoids gRPC handshake drift)
    if (!this.streamOpenedAt) {
      this.streamOpenedAt = new Date();
    }
    try {
      // SDK v6 auto-wraps raw buffers in { audioContent } via its pipeline
      this.recognizeStream.write(pcm);
    } catch {
      // Stream may have been closed by server
      this._open = false;
    }
  }

  close(): void {
    if (!this._open) return;
    this._open = false;
    // Graceful close — any pending interims will be flushed in the 'end' handler.
    try {
      this.recognizeStream?.end();
    } catch {
      // Already closed
    }
  }

  get open(): boolean {
    return this._open;
  }
}
