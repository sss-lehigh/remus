# Scratch Notes

```mermaid
graph TD
    B(broker.h) --> D
    B --> RCV
    CH(channel.h) --> MSG
    CM(connection_manager.h) --> CH
    CM --> MSG
    CM --> RCV
    CM --> B
    D(device.h)
    MEM(memory.h)
    MP(memory_pool.h) --> CH
    MP --> CM
    MP --> MSG
    MP --> RM
    MSG(messenger.h) --> MEM
    RCV(receiver.h)
    RM(rmalloc.h) --> MEM
```
