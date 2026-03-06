import type { SttEngine, SttStream, TranscriptCallback, SttConfig } from './types.js';
import { logger } from '../logger.js';

/**
 * Whisper engine stub for offline batch transcription.
 * Not used for real-time streaming — only for post-session re-transcription.
 */
export class WhisperEngine implements SttEngine {
  private config: SttConfig;

  constructor(config: SttConfig) {
    this.config = config;
    logger.info('Whisper engine initialized (batch mode only)');
  }

  createStream(_userId: string, _streamSequence: number, _onTranscript: TranscriptCallback): SttStream {
    throw new Error('WhisperEngine does not support real-time streaming. Use for batch transcription only.');
  }

  /**
   * Batch-process an OGG audio file.
   * Returns transcript segments with timestamps.
   */
  async transcribeFile(_filePath: string, _userId: string): Promise<Array<{
    segmentStart: string;
    segmentEnd: string;
    transcript: string;
    confidence: number;
  }>> {
    // TODO: Implement whisper.cpp bindings
    // 1. Decode OGG -> WAV via ffmpeg
    // 2. Run whisper.cpp with --output-json
    // 3. Parse timestamp + text from JSON output
    throw new Error('Whisper transcription not yet implemented. Install whisper-node and configure stt.whisper.modelPath.');
  }

  destroy(): void {
    // No resources to clean up for batch mode
  }
}
