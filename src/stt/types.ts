import type { Readable } from 'stream';

export interface TranscriptEvent {
  userId: string;
  displayName: string | null;
  speakerLabel: string | null;
  segmentStart: string;
  segmentEnd: string | null;
  transcript: string;
  confidence: number | null;
  isFinal: boolean;
  sttResultId: string | null;
  streamSequence: number;
  sttEngine: string;
  sttModel: string | null;
}

export interface SttStream {
  write(pcm: Buffer): void;
  close(): void;
  readonly open: boolean;
  /** Called when the stream closes unexpectedly (server error / duration limit). */
  onClose?: () => void;
}

export type TranscriptCallback = (event: TranscriptEvent) => void;

export interface SttEngine {
  createStream(userId: string, streamSequence: number, onTranscript: TranscriptCallback): SttStream;
  destroy(): void;
}

/**
 * Abstract PCM audio source — anything that produces a readable stream of PCM data.
 * Decouples VadGate from Discord-specific FfmpegResampler.
 */
export interface PcmSource {
  readonly pcmOutput: Readable;
}

export interface SttConfig {
  enabled: boolean;
  engine: 'google-cloud-stt' | 'whisper' | 'gemini';
  googleCloud: {
    projectId: string;
    keyFile: string;
    model: string;
    languageCode: string;
    enableAutomaticPunctuation: boolean;
    sampleRateHertz: number;
    streamRotationMinutes: number;
    streamOverlapSeconds: number;
    phraseHints: string[];
  };
  diarization: { minSpeakers: number; maxSpeakers: number };
  silenceTimeoutSeconds: number;
  connectionCooldownSeconds: number;
  whisper: { modelPath: string; language: string };
  costWarningPerSessionUsd: number;
  maxConcurrentStreams: number;
  interimThrottlePerSecond: number;
}
