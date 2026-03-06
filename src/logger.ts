export interface Logger {
  debug(...args: unknown[]): void;
  info(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
}

const noopLogger: Logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

let currentLogger: Logger = noopLogger;

/**
 * Set the logger implementation used by all @magi/common modules.
 * Call this once at startup before using any STT classes.
 */
export function setLogger(l: Logger): void {
  currentLogger = l;
}

/** Proxy that delegates to the current logger set via setLogger(). */
export const logger: Logger = {
  debug: (...args) => currentLogger.debug(...args),
  info: (...args) => currentLogger.info(...args),
  warn: (...args) => currentLogger.warn(...args),
  error: (...args) => currentLogger.error(...args),
};
