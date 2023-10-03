import { Metric } from "./metrics";

// [mfs] This needs documentation

// [mfs]  I don't understand why the variant Metric is being used.  Is ops
//        really able to be a Counter/Stopwatch/Summary, or should this be more
//        granular?
export class WorkloadDriver {
  ops?: Metric;
  runtime?: Metric;
  qps?: Metric;
  latency?: Metric;
};