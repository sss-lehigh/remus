// [mfs] This file needs better documentation

export class Metric {
  name?: string;
  metric: Counter | Stopwatch | Summary;
};

class Counter {
  /** A counter (uint64_t) */
  count?: number;
};

class Stopwatch {
  /** A running time, in nanoseconds (uint64_t) */
  runtime_ns?: number;
};

class Summary {
  /** Measurement units */
  units?: string;

  // Summary statistics (all doubles)
  mean?: number;
  stddev?: number;
  min?: number;
  p50?: number;
  p90?: number;
  p95?: number;
  p99?: number;
  p999?: number;
  max?: number;

  /** Total number of samples collected (uint64_t) */
  count?: number;
};