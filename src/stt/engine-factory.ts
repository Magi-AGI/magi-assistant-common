import type { SttEngine, SttConfig } from './types.js';
import { GoogleCloudSttEngine } from './google-cloud-engine.js';
import { GoogleCloudDiarizedEngine } from './google-cloud-diarized-engine.js';
import { WhisperEngine } from './whisper-engine.js';
import { logger } from '../logger.js';

interface EngineEntry {
  engine: SttEngine;
  refCount: number;
}

/**
 * Singleton engine registry, keyed by engine type + diarized mode.
 * Prevents multi-guild sessions from destroying each other's gRPC clients.
 */
const engines = new Map<string, EngineEntry>();

function makeKey(config: SttConfig, diarized: boolean): string {
  return `${config.engine}:${diarized}`;
}

function createEngine(config: SttConfig, diarized: boolean): SttEngine {
  switch (config.engine) {
    case 'google-cloud-stt':
      if (diarized) {
        return new GoogleCloudDiarizedEngine(config);
      }
      return new GoogleCloudSttEngine(config);
    case 'whisper':
      return new WhisperEngine(config);
    default:
      throw new Error(`Unknown STT engine: ${config.engine}`);
  }
}

/**
 * Acquire a reference to an STT engine. Reuses existing instance if one
 * is already active for the same engine type + diarized mode.
 */
export function acquireSttEngine(config: SttConfig, diarized: boolean = false): SttEngine {
  const key = makeKey(config, diarized);
  const existing = engines.get(key);
  if (existing) {
    existing.refCount++;
    logger.debug(`STT engine acquired (reuse): ${key}, refCount=${existing.refCount}`);
    return existing.engine;
  }

  const engine = createEngine(config, diarized);
  engines.set(key, { engine, refCount: 1 });
  logger.debug(`STT engine acquired (new): ${key}`);
  return engine;
}

/**
 * Release a reference to an STT engine. Destroys the engine when
 * the last session releases it.
 */
export function releaseSttEngine(config: SttConfig, diarized: boolean = false): void {
  const key = makeKey(config, diarized);
  const entry = engines.get(key);
  if (!entry) return;

  entry.refCount--;
  logger.debug(`STT engine released: ${key}, refCount=${entry.refCount}`);

  if (entry.refCount <= 0) {
    entry.engine.destroy();
    engines.delete(key);
    logger.debug(`STT engine destroyed: ${key}`);
  }
}

/** Destroy all engines (for shutdown). */
export function destroyAllEngines(): void {
  for (const [key, entry] of engines) {
    entry.engine.destroy();
    logger.debug(`STT engine force-destroyed: ${key}`);
  }
  engines.clear();
}
