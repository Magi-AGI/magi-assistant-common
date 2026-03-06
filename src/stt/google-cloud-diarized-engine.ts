import { SpeechClient, protos } from '@google-cloud/speech';
import type { SttEngine, SttStream, TranscriptCallback, TranscriptEvent, SttConfig } from './types.js';
import { logger } from '../logger.js';

type IStreamingRecognizeResponse = protos.google.cloud.speech.v1.IStreamingRecognizeResponse;

/**
 * Google Cloud STT engine with speaker diarization enabled.
 * Used when --diarized flag is set — single GM stream, multiple speakers identified.
 */
export class GoogleCloudDiarizedEngine implements SttEngine {
  private client: SpeechClient;
  private config: SttConfig;

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
    logger.info('Google Cloud STT diarized engine initialized');
  }

  createStream(userId: string, streamSequence: number, onTranscript: TranscriptCallback): SttStream {
    return new DiarizedSttStream(
      this.client,
      this.config,
      userId,
      streamSequence,
      onTranscript
    );
  }

  destroy(): void {
    this.client.close().catch((err: unknown) => {
      logger.warn('Error closing Google Cloud STT diarized client:', err);
    });
  }
}

class DiarizedSttStream implements SttStream {
  private recognizeStream: ReturnType<SpeechClient['streamingRecognize']> | null = null;
  private _open = true;
  /** Set on first audio write (not constructor) to avoid gRPC handshake latency drift. */
  private streamOpenedAt: Date | null = null;
  private lastResultEndMs = 0;
  onClose?: () => void;

  constructor(
    client: SpeechClient,
    private config: SttConfig,
    private gmUserId: string,
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
        diarizationConfig: {
          enableSpeakerDiarization: true,
          minSpeakerCount: config.diarization.minSpeakers,
          maxSpeakerCount: config.diarization.maxSpeakers,
        },
      },
      interimResults: true,
    };

    this.recognizeStream = client.streamingRecognize(streamingConfig);

    this.recognizeStream.on('data', (response: IStreamingRecognizeResponse) => {
      if (!response.results || response.results.length === 0) return;

      for (const result of response.results) {
        if (!result.alternatives || result.alternatives.length === 0) continue;

        const alt = result.alternatives[0];
        const isFinal = result.isFinal === true;

        // Extract speaker tag from word-level results
        let speakerTag: number | null = null;
        if (alt.words && alt.words.length > 0) {
          const lastWord = alt.words[alt.words.length - 1];
          if (typeof lastWord.speakerTag === 'number' && lastWord.speakerTag > 0) {
            speakerTag = lastWord.speakerTag;
          }
        }

        const speakerLabel = speakerTag !== null ? `speaker_${speakerTag}` : null;

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

        const event: TranscriptEvent = {
          userId: `diarized:${this.gmUserId}`,
          displayName: null,
          speakerLabel,
          segmentStart,
          segmentEnd,
          transcript: alt.transcript ?? '',
          confidence: typeof alt.confidence === 'number' ? alt.confidence : null,
          isFinal,
          sttResultId: result.resultEndTime
            ? `${result.resultEndTime.seconds ?? 0}_${result.resultEndTime.nanos ?? 0}`
            : null,
          streamSequence: this.streamSequence,
          sttEngine: 'google-cloud-stt',
          sttModel: config.googleCloud.model,
        };

        this.onTranscript(event);
      }
    });

    this.recognizeStream.on('error', (err: Error) => {
      if ('code' in err && (err as { code: number }).code === 11) {
        logger.debug(`Google STT diarized stream ended (duration limit) for GM ${gmUserId}`);
      } else {
        logger.error(`Google STT diarized stream error for GM ${gmUserId}:`, err);
      }
      this._open = false;
      this.onClose?.();
    });

    this.recognizeStream.on('end', () => {
      this._open = false;
      this.onClose?.();
    });
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
      this._open = false;
    }
  }

  close(): void {
    if (!this._open) return;
    this._open = false;
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
