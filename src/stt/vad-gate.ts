import { EventEmitter } from 'events';
import type { SttEngine, SttStream, TranscriptEvent, PcmSource, SttConfig } from './types.js';
import { logger } from '../logger.js';

/**
 * VAD-gated STT stream manager for a single user.
 * Uses `receiver.speaking` events (forwarded from SttProcessor) to open/close STT streams.
 * Handles stream rotation (4 min cumulative speech) with overlap-based dedup.
 */
export class VadGate extends EventEmitter {
  private userId: string;
  private engine: SttEngine;
  private pcmSource: PcmSource;
  private config: SttConfig;

  private currentStream: SttStream | null = null;
  private streamSequence = 0;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private cooldownUntil = 0;
  private cumulativeSpeechMs = 0;
  private speechStartTime: number | null = null;
  private destroyed = false;
  private cooldownReopenTimer: ReturnType<typeof setTimeout> | null = null;
  private rotationCheckTimer: ReturnType<typeof setInterval> | null = null;

  // For rotation dedup
  private rotatingStream: SttStream | null = null;
  private rotationTimer: ReturnType<typeof setTimeout> | null = null;
  private lastNewStreamFirstResult: string | null = null;

  /** Bound PCM data handler — stored so we can unbind it in destroy(). */
  private boundPcmHandler: (pcm: Buffer) => void;

  constructor(userId: string, engine: SttEngine, pcmSource: PcmSource, config: SttConfig) {
    super();
    this.userId = userId;
    this.engine = engine;
    this.pcmSource = pcmSource;
    this.config = config;

    // Pipe PCM output to active STT stream
    this.boundPcmHandler = (pcm: Buffer) => {
      if (this.destroyed) return;
      if (this.currentStream?.open) {
        this.currentStream.write(pcm);
      }
      if (this.rotatingStream?.open) {
        this.rotatingStream.write(pcm);
      }
    };
    this.pcmSource.pcmOutput.on('data', this.boundPcmHandler);
  }

  onSpeakingStart(): void {
    if (this.destroyed) return;

    // Cancel silence timer
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }

    // Track speech start for cumulative timing
    this.speechStartTime = Date.now();

    // Start periodic rotation check for sustained speech
    // (handles cases where speaking doesn't pause at all for > rotationMinutes)
    if (!this.rotationCheckTimer) {
      const checkIntervalMs = 30_000; // check every 30s
      this.rotationCheckTimer = setInterval(() => {
        if (this.destroyed) return;
        // Accumulate current speaking burst without resetting speechStartTime
        if (this.speechStartTime) {
          const runningMs = this.cumulativeSpeechMs + (Date.now() - this.speechStartTime);
          const rotationMs = this.config.googleCloud.streamRotationMinutes * 60 * 1000;
          if (runningMs >= rotationMs && this.currentStream?.open) {
            // Emit speech duration for the pre-rotation portion before resetting
            const elapsedMs = Date.now() - this.speechStartTime;
            this.emit('speechDuration', elapsedMs);

            // Finalize current accumulation
            this.cumulativeSpeechMs = runningMs;
            this.speechStartTime = Date.now(); // restart timing from now
            this.rotateStream();
          }
        }
      }, checkIntervalMs);
    }

    // Cancel any pending cooldown reopen
    if (this.cooldownReopenTimer) {
      clearTimeout(this.cooldownReopenTimer);
      this.cooldownReopenTimer = null;
    }

    // Open stream if none active (with cooldown check)
    if (!this.currentStream?.open) {
      const now = Date.now();
      if (now < this.cooldownUntil) {
        // Schedule a deferred open after cooldown expires
        const delayMs = this.cooldownUntil - now;
        logger.debug(`VadGate (${this.userId}): cooldown active, scheduling reopen in ${delayMs}ms`);
        this.cooldownReopenTimer = setTimeout(() => {
          this.cooldownReopenTimer = null;
          if (!this.destroyed && !this.currentStream?.open) {
            this.openNewStream();
          }
        }, delayMs);
        return;
      }

      this.openNewStream();
    }
  }

  private openNewStream(): void {
    this.streamSequence++;
    this.cumulativeSpeechMs = 0;

    this.currentStream = this.engine.createStream(
      this.userId,
      this.streamSequence,
      (event) => this.handleTranscript(event)
    );

    // Listen for unexpected stream close (gRPC error / duration limit).
    // If the user is mid-speech, attempt to reopen after cooldown expires.
    this.currentStream.onClose = () => {
      if (this.destroyed) return;
      // Only reopen if this is still the current stream (not a stale rotation target)
      if (this.currentStream && !this.currentStream.open && this.speechStartTime !== null) {
        logger.warn(`VadGate (${this.userId}): stream #${this.streamSequence} closed unexpectedly mid-speech, reopening`);
        this.currentStream = null;
        const cooldownMs = this.config.connectionCooldownSeconds * 1000;
        this.cooldownUntil = Date.now() + cooldownMs;
        // Defer reopen until cooldown expires (stored for cleanup in destroy())
        this.cooldownReopenTimer = setTimeout(() => {
          this.cooldownReopenTimer = null;
          if (!this.destroyed && this.speechStartTime !== null && !this.currentStream?.open) {
            this.openNewStream();
          }
        }, cooldownMs);
      }
    };

    logger.debug(`VadGate (${this.userId}): stream #${this.streamSequence} opened`);
  }

  onSpeakingEnd(): void {
    if (this.destroyed) return;

    // Stop periodic rotation check (will restart on next speaking start)
    if (this.rotationCheckTimer) {
      clearInterval(this.rotationCheckTimer);
      this.rotationCheckTimer = null;
    }

    // Cancel any pending cooldown reopen (speech ended, so no need to reopen)
    if (this.cooldownReopenTimer) {
      clearTimeout(this.cooldownReopenTimer);
      this.cooldownReopenTimer = null;
    }

    // Accumulate speech duration and emit for usage tracking
    if (this.speechStartTime) {
      const burstDurationMs = Date.now() - this.speechStartTime;
      this.cumulativeSpeechMs += burstDurationMs;
      this.speechStartTime = null;
      this.emit('speechDuration', burstDurationMs);
    }

    // Check if rotation needed (4 min cumulative speech)
    this.checkRotation();

    // Start silence timer
    this.silenceTimer = setTimeout(() => {
      this.closeCurrentStream();
    }, this.config.silenceTimeoutSeconds * 1000);
  }

  private checkRotation(): void {
    const rotationMs = this.config.googleCloud.streamRotationMinutes * 60 * 1000;
    if (this.cumulativeSpeechMs >= rotationMs && this.currentStream?.open) {
      this.rotateStream();
    }
  }

  private rotateStream(): void {
    logger.debug(`VadGate (${this.userId}): rotating stream after ${this.cumulativeSpeechMs}ms cumulative speech`);

    // If a previous rotation is still in the overlap period, close it immediately
    // to prevent leaked gRPC streams from overwriting the reference.
    if (this.rotatingStream?.open) {
      this.rotatingStream.close();
      logger.debug(`VadGate (${this.userId}): force-closed previous rotating stream (back-to-back rotation)`);
    }
    if (this.rotationTimer) {
      clearTimeout(this.rotationTimer);
      this.rotationTimer = null;
    }

    // Old stream becomes the rotating stream (will be closed after overlap)
    this.rotatingStream = this.currentStream;

    // Open new stream
    this.streamSequence++;
    this.cumulativeSpeechMs = 0;
    this.lastNewStreamFirstResult = null;

    this.currentStream = this.engine.createStream(
      this.userId,
      this.streamSequence,
      (event) => this.handleTranscript(event)
    );

    // Set onClose for mid-speech recovery (same as openNewStream)
    this.currentStream.onClose = () => {
      if (this.destroyed) return;
      if (this.currentStream && !this.currentStream.open && this.speechStartTime !== null) {
        logger.warn(`VadGate (${this.userId}): rotated stream #${this.streamSequence} closed unexpectedly mid-speech, reopening`);
        this.currentStream = null;
        const cooldownMs = this.config.connectionCooldownSeconds * 1000;
        this.cooldownUntil = Date.now() + cooldownMs;
        this.cooldownReopenTimer = setTimeout(() => {
          this.cooldownReopenTimer = null;
          if (!this.destroyed && this.speechStartTime !== null && !this.currentStream?.open) {
            this.openNewStream();
          }
        }, cooldownMs);
      }
    };

    // Close old stream after overlap period
    const overlapMs = this.config.googleCloud.streamOverlapSeconds * 1000;
    this.rotationTimer = setTimeout(() => {
      if (this.rotatingStream?.open) {
        this.rotatingStream.close();
      }
      this.rotatingStream = null;
      this.rotationTimer = null;
    }, overlapMs);
  }

  private closeCurrentStream(): void {
    if (this.currentStream?.open) {
      this.currentStream.close();
      this.cooldownUntil = Date.now() + this.config.connectionCooldownSeconds * 1000;
      logger.debug(`VadGate (${this.userId}): stream #${this.streamSequence} closed (silence timeout)`);
    }
    this.currentStream = null;
  }

  private handleTranscript(event: TranscriptEvent): void {
    if (this.destroyed) return;

    // Dedup: if we're rotating and this is from the old stream,
    // discard if its timestamp falls within +/-2s of the new stream's first result
    if (this.rotatingStream && event.streamSequence < this.streamSequence) {
      if (this.lastNewStreamFirstResult) {
        const eventTime = new Date(event.segmentStart).getTime();
        const newStreamTime = new Date(this.lastNewStreamFirstResult).getTime();
        if (Math.abs(eventTime - newStreamTime) <= 2000) {
          return; // Within +/-2s dedup window -- drop old-stream result
        }
      }
    }

    // Track first result from new stream for dedup
    if (event.streamSequence === this.streamSequence && !this.lastNewStreamFirstResult) {
      this.lastNewStreamFirstResult = event.segmentStart;
    }

    this.emit('transcript', event);
  }

  destroy(): void {
    this.destroyed = true;

    // Emit remaining speech duration if user was mid-speech (prevents usage under-count)
    if (this.speechStartTime) {
      const remainingMs = Date.now() - this.speechStartTime;
      this.speechStartTime = null;
      this.emit('speechDuration', remainingMs);
    }

    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }

    if (this.cooldownReopenTimer) {
      clearTimeout(this.cooldownReopenTimer);
      this.cooldownReopenTimer = null;
    }

    if (this.rotationCheckTimer) {
      clearInterval(this.rotationCheckTimer);
      this.rotationCheckTimer = null;
    }

    if (this.rotationTimer) {
      clearTimeout(this.rotationTimer);
      this.rotationTimer = null;
    }

    if (this.currentStream?.open) {
      this.currentStream.close();
    }
    if (this.rotatingStream?.open) {
      this.rotatingStream.close();
    }

    this.currentStream = null;
    this.rotatingStream = null;

    // Explicitly unbind from PCM source stream to prevent zombie listeners
    // (VadGate is transient — recreated on rejoin/rebind — but pcmSource is long-lived)
    this.pcmSource.pcmOutput.off('data', this.boundPcmHandler);

    this.removeAllListeners();
  }
}
