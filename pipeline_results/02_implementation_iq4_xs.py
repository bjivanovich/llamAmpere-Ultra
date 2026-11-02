We need answer user in English. Need produce final with detailed Python architecture. Need include code. Need decide final design.

Let's design robust code.

We'll produce a single file maybe `pipeline.py`.

Types:
```
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
```

Task protocol:
```
@dataclass(slots=True)
class Task:
    task_id: str
    kind: str
    payload: Any
    created_at: float
```
But queue item maybe Task.

Result:
```
@dataclass(slots=True)
class TaskResult:
    task_id: str
    status: TaskStatus
    value: Any
    error: BaseException | None
    started: float
    finished: float
```

WAL record:
```
@dataclass(slots=True)
class WALRecord:
    seq: int
    task_id: str
    kind: str
    payload: Any
    created_at: float
```
Maybe store task.

WAL file:
Need binary serialization. Use pickle? Production: use pickle with `protocol=5`, with checksum (zlib.crc32) and magic. Need thread-safe.

Record format:
MAGIC 4 bytes, VERSION 1, SEQ 8, PAYLOAD_LEN 8, PAYLOAD, CRC32 4.
```
HEADER = b'ATXWAL01'
```
Write:
```
with self._lock:
   if closed: raise
   seq = next
   data = pickle.dumps(payload, protocol=5)
   frame = MAGIC + bytes([VERSION]) + seq.to_bytes(8) + len.to_bytes(8) + data + crc.to_bytes(8)
   for chunk in os.write(fd, frame): ...
   if fsync: os.fsync(fd)
```
Need thread-safe: `threading.Lock`.
Need `async` method? Provide `await write(...)` using loop.run_in_executor.

WAL class:
```
class WAL:
    def __init__(self, path, fsync_interval=1, max_buffer=1024*1024)
      fd=os.open(path, O_WRONLY|O_CREAT|O_APPEND, 0o644)
      lock=threading.Lock()
      seq=0
      pending=0
      _fsync_lock?
```
Simpler: lock for writes, fsync lock for fsync.
Write:
```
def write(self, record):
  with self._write_lock:
    if closed: raise
    seq = self._seq; self._seq +=1
    data = pickle.dumps(record.payload, protocol=5)
    frame = ...
    view=memoryview(frame)
    while view:
       n=os.write(fd, view)
       if n==-1: if EINTR continue; raise
       view = view[n:]
    self._pending += 1
    if self._pending % fsync_interval == 0:
       self._fsync()
```
But if multiple threads write, fsync on one. Need fsync lock.

`fsync` method:
```
with self._fsync_lock:
  if os.fsync(fd) == -1: ...
```
Need close:
```
with lock:
  self._closed=True
  os.close(fd)
  self._fsync? before close.
```
Need `replay` classmethod:
```
@classmethod
def replay(cls, path, limit=None):
  open rb
  records=[]
  while True:
    header=read(21?)
    if not header: break
    if len(header)<21: if len(header)>0 raise
    if header[0:4]!=MAGIC raise
    ver=header[4]
    seq=int.from_bytes(header[5:13])
    plen=int.from_bytes(header[13:21])
    data=read(plen); crc=read(4)
    if crc mismatch raise
    payload=pickle.loads(data)
    records.append(payload)
```
Need maybe `os.fork`? Not.

State Cache:
Memory-efficient with OrderedDict. Need TTL and LRU. Use threading.RLock? For thread-safe. Need `max_size`, `ttl`, `clock=time.monotonic`, `max_bytes` maybe. Could implement:
```
class StateCache(Generic[K,V]):
    def __init__(self, max_size, ttl, max_bytes=0, now=time.monotonic):
       self._d=OrderedDict()
       self._lock=threading.RLock()
       self._hits=0; self._misses=0; self._evictions=0; self._expired=0; self._size=0; self._bytes=0
    def _key(k): return k
    def put(key,value,ttl=None):
       with lock:
         if key in d: remove
         exp = now()+ttl if ttl else now()+self.ttl
         d[key]=(value, exp)
         if max_size and len>max_size: popitem(last=False)
         if max_bytes and size>max_bytes: pop while
         return True
    def get(key, default=MISSING):
       with lock:
         entry=d.get(key)
         if entry is None: miss
         if now()>exp: del; expired++
         else move_to_end; hits++
    def delete(key):
    def sweep():
       with lock:
         for k in list(d): if expired del
    def close():
       with lock: clear
```
Need thread-safe. Use `move_to_end`.
Potential issue: `__missing__`? no.
Need `__len__`, `__contains__`, `stats`.

Could also support `get_or_set`? Not necessary.

Need `MISSING = object()`.

BoundedQueue:
Use Event implementation.
```
class BoundedQueue(Generic[T]):
   def __init__(self, maxlen:int):
      if maxlen<0: raise
      self._items=deque()
      self._maxlen=maxlen
      self._not_empty=asyncio.Event()
      self._not_full=asyncio.Event()
      self._not_full.set()
      self._closed=False
      self._put_count=0; self._get_count=0; self._unfinished=0
```
Need `put`:
```
async def put(self, item):
   if self._maxlen == 0:
      if item is None? no.
   if self._closed: raise QueueClosed
   while True:
      if self._closed: raise
      if len(self._items) >= self._maxlen:
         self._not_full.clear()
