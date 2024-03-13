/**
 * RemoteObject represents a remotely accessible memory region.  It is used to
 * convey the necessary information for remote nodes to interact with this
 * memory, assuming that they have access to a QP that is connected to it.
 *
 * [mfs]  IHT seems to only use raddr.  Remus also uses rkey.  Do we need the
 *        rest?
 */
export class RemoteObject {
  /** An identifier for this object. Must be unique among remote objects */
  id?: string;

  /** Address of first byte in the memory region (uint64_t) */
  raddr?: number;

  /** Size of the memory region (uint32_t) */
  size?: number;

  /** Local access key (uint32_t) */
  lkey?: number;

  /** Remote access key (uint32_t) */
  rkey?: number;
}