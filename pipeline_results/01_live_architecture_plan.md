<think>
The user is asking me to design a complete technical architecture for a real-time distributed network telemetry engine in Python. Key requirements:

1. Distributed network telemetry engine
2. Real-time
3. Python
4. Copy-free shared circular buffers (ring buffers)
5. Async loop with multiprocessing

Let me design this rigorously, covering:
- Module decomposition
- Protocols between components
- Memory structures (ring buffers)
- API contracts
- Async/multiprocessing architecture
- Data flow
- Synchronization primitives
- Fault tolerance
- Performance considerations

I need to be exhaustive and rigorous. Let me think about the architecture:

**Core Components:**
1. Ingestion Layer - captures network telemetry (packets, metrics)
2. Ring Buffer Layer - shared memory circular buffers (zero-copy)
3. Processing Engine - async workers that consume from ring buffers
4. Distribution Layer - distributes processed telemetry to consumers
5. Control Plane - coordination, health, backpressure
6. Storage/Export - sinks for processed data

**Key Design Decisions:**
- Use `multiprocessing.shared_memory` for ring buffers (Python 3.8+)
- `asyncio` for the async event loop
- `multiprocessing.Process` for CPU-bound workers
- Lock-free ring buffer implementation (single producer, multiple consumer pattern)
- Memory-mapped files as fallback
- `os.pipe` or `eventfd` for cross-process notification

**Ring Buffer Design:**
- Fixed-size circular buffer in shared memory
- Header: write index, read index, sequence number, magic bytes
- Payload: fixed-size telemetry records
- SPSC (Single Producer Single Consumer) for lock-free operation
- SPMC (Single Producer Multiple Consumer) with per-consumer read pointers

**Async Architecture:**
- Main process: asyncio event loop
- Worker processes: each with their own asyncio loop
- Communication: ring buffers (data plane) + pipes/queues (control plane)
- Backpressure: watermarks on ring buffers

Let me structure this comprehensively with code-level detail.
</think>

# Arquitectura: Motor Distribuido de Telemetría de Red en Tiempo Real

## 1. Visión General y Principios de Diseño

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CONTROL PLANE (gRPC / Unix Domain Socket)             │
│   ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐   │
│   │ Orchestr.│  │ Health/Heart │  │ Backpressure│  │ Config Hot-Reload    │   │
│   └──────────┘  └──────────────┘  └─────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                              │  ▲  │  ▲
                              ▼  │  ▼  │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PLANE                                         │
│                                                                                 │
│  ┌───────────┐   ┌─────────────────────────────────────────────────────────┐   │
│  │  INGEST   │   │              SHARED RING BUFFER POOL (mmap)             │   │
│  │  (NIF)    │──▶│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
│  │  proc 0   │   │  │ Ring 0  │ │ Ring 1  │ │ Ring 2  │ │ Ring N  │     │   │
│  └───────────┘   │  │ SPSC    │ │ SPMC    │ │ SPSC    │ │ SPMC    │     │   │
│  ┌───────────┐   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │   │
│  │  INGEST   │   │       │           │           │           │          │   │
│  │  (AF_XDP) │   └───────┼───────────┼───────────┼───────────┼──────────┘   │
│  │  proc 1   │           │           │           │           │              │
│  └───────────┘           ▼           ▼           ▼           ▼              │
│                        ┌────────────────────────────────────────────────┐    │
│                        │         ASYNC WORKER POOL (multiproc)          │    │
│                        │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │    │
│                        │  │Worker 0│ │Worker 1│ │Worker 2│ │Worker N│  │    │
│                        │  │asyncio │ │asyncio │ │asyncio │ │asyncio │  │    │
│                        │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘  │    │
│                        └──────┼──────────┼──────────┼──────────┼────────┘    │
│                               ▼          ▼          ▼          ▼             │
│                        ┌────────────────────────────────────────────────┐    │
│                        │         DISTRIBUTION / EXPORT LAYER            │    │
│                        │   Kafka │ gRPC │ Redis │ File │ Custom Sink    │    │
│                        └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Principios inmutables:**

| Principio | Implementación |
|-----------|---------------|
| Zero-copy en data plane | `multiprocessing.shared_memory` + `memoryview` |
| Lock-free en hot path | SPSC con `atomic` via `struct.pack/unpack` + `sys.byteorder` |
| Backpressure | Watermarks + `eventfd`/pipe señalización |
| Fault isolation | Cada worker en proceso independiente |
| Deterministic latency | Bounded buffers, no GC en hot path (`gc.freeze()`) |

---

## 2. Estructura de Memoria: Ring Buffer Compartido

### 2.1 Layout en Memoria Compartida

```python
# ring_buffer_layout.py
"""
Layout binario de un ring buffer en shared memory.
Tamaño total = HEADER_SIZE + (SLOT_COUNT * SLOT_SIZE)

HEADER (64 bytes, alineado a 64B para cache-line):
  [0:4]    magic:       b'RTLB'
  [4:8]    version:     uint32  = 1
  [8:16]   seq_write:   uint64  (monotonic, producer)
  [16:24]  seq_read_min: uint64 (min de todos los consumers)
  [24:32]  seq_read_max: uint64 (max de todos los consumers)
  [32:40]  watermark_high: uint64 (umbral de backpressure)
  [40:48]  watermark_low:  uint64 (umbral de recuperación)
  [48:56]  slot_size:    uint64  (tamaño fijo de cada slot)
  [56:64]  slot_count:   uint64  (debe ser potencia de 2)
  [64:80]  consumer_ptrs[0]: uint64  (read cursor del consumer 0)
  [80:96]  consumer_ptrs[1]: uint64
  ...      consumer_ptrs[i]: uint64  (hasta MAX_CONSUMERS)

SLOT (tamaño fijo, ej. 256 bytes):
  [0:4]    seq:         uint64  (secuencia del record, para detección de huecos)
  [4:8]    ts_ns:       uint64  (timestamp de captura, CLOCK_MONOTONIC)
  [8:12]   src_ip:      uint32
  [12:16]  dst_ip:      uint32
  [16:18]  src_port:    uint16
  [18:20]  dst_port:    uint16
  [20:21]  protocol:    uint8   (6=TCP, 17=UDP)
  [21:22]  flags:       uint8   (bit0: SYN, bit1: ACK, bit2: FIN, bit3: RST)
  [22:24]  pkt_len:     uint16
  [24:28]  if_index:    uint32
  [28:32]  vlan_id:     uint32
  [32:64]  metadata:    uint8[32]  (BPF tags, flow_id, etc.)
  [64:256] payload:     uint8[192]