import { WorkloadDriver } from "./colosseum";

// [mfs] This file needs more documentation

class Ack { };

class ExperimentParams {
  // int32
  think_time: number = 0;

  // int32
  qps_sample_rate: number = 10;

  // int32
  max_qps_second: number = -1;

  // int32
  runtime: number = 10;

  unlimited_stream: boolean = false;

  // int32
  op_count: number = 10000;

  // int32
  contains: number = 80;

  // int32
  insert: number = 10;

  // int32
  remove: number = 10;

  // int32
  key_lb: number = 0;

  // int32
  key_ub: number = 1000000;

  // int32
  region_size: number = 22;

  // int32
  thread_count: number = 1;

  // int32
  node_count: number = 0;
}

class Result {
  params?: ExperimentParams;
  driver: WorkloadDriver[];
}