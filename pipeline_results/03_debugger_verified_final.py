#!/usr/bin/env python3
"""
Real-Time In-Memory Time-Series Telemetry & Analytics Engine
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

# --- Data Structures ---

@dataclass(frozen=True)
class TelemetryRecord:
    timestamp: float
    metric: str
    value: float
    tags: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(f"timestamp must be numeric, got {type(self.timestamp)}")
        if not isinstance(self.metric, str):
            raise TypeError(f"metric must be str, got {type(self.metric)}")
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"value must be numeric, got {type(self.value)}")
        if not self.tags:
            object.__setattr__(self, 'tags', tuple())
        else:
            object.__setattr__(self, 'tags', tuple(self.tags))


class RingBuffer:
    """Lock-free-ish circular buffer with pre-allocated arrays."""
    
    __slots__ = ('capacity', '_times', '_values', '_metrics', '_tags',
                  '_pos', '_count', '_lock', '_overflow')
    
    def __init__(self, capacity: int = 10_000):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._times = [0.0] * capacity
        self._values = [0.0] * capacity
        self._metrics = [''] * capacity
        self._tags = [()] * capacity
        self._pos = 0
        self._count = 0
        self._lock = threading.Lock()
        self._overflow = 0
    
    def append(self, rec: TelemetryRecord) -> bool:
        with self._lock:
            if self._count == self.capacity:
                self._overflow += 1
                self._pos = (self._pos + 1) % self.capacity
            self._times[self._pos] = rec.timestamp
            self._values[self._pos] = rec.value
            self._metrics[self._pos] = rec.metric
            self._tags[self._pos] = rec.tags
            self._pos = (self._pos + 1) % self.capacity
            self._count += 1
            return True
    
    def snapshot(self, metric: Optional[str] = None,
                 window: Optional[float] = None) -> List[Tuple[float, str, float, tuple]]:
        with self._lock:
            n = self._count
            if n == 0:
                return []
            start = (self._pos - n) % self.capacity
            out = []
            for i in range(n):
                idx = (start + i) % self.capacity
                t = self._times[idx]
                m = self._metrics[idx]
                v = self._values[idx]
                tg = self._tags[idx]
                if metric is not None and m != metric:
                    continue
                if window is not None and t < (self._times[start] - window):
                    continue
                out.append((t, m, v, tg))
            return out
    
    def __len__(self) -> int:
        with self._lock:
            return self._count
    
    @property
    def overflow(self) -> int:
        with self._lock:
            return self._overflow


# --- Analytics ---

class EWMA:
    __slots__ = ('alpha', 'mean', 'var', 'n')
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.mean = float('nan')
        self.var = float('nan')
        self.n = 0
    
    def update(self, x: float) -> Tuple[float, float, int]:
        a = self.alpha
        if self.n == 0:
            self.mean = x
            self.var = 0.0
            self.n = 1
        else:
            diff = x - self.mean
            self.mean += a * diff
            self.var = (1 - a) * self.var + a * (diff * diff)
            if self.var < 0:
                self.var = 0.0
            self.n += 1
        return self.mean, self.var, self.n


class QuantileTracker:
    __slots__ = ('values',)
    
    def __init__(self):
        self.values: List[float] = []
    
    def update(self, x: float) -> None:
        self.values.append(x)
    
    def snapshot(self) -> Dict[str, float]:
        vals = sorted(self.values)
        n = len(vals)
        if n == 0:
            return {'p50': float('nan'), 'p95': float('nan'), 'p99': float('nan'),
                    'min': float('nan'), 'max': float('nan'),
                    'mean': float('nan'), 'std': float('nan'), 'n': 0}
        
        def _pct(q: float) -> float:
            idx = max(0, min(n - 1, int(math.ceil(q * n)) - 1))
            return vals[idx]
        
        mean = sum(vals) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        
        return {
            'p50': _pct(0.50),
            'p95': _pct(0.95),
            'p99': _pct(0.99),
            'min': vals[0],
            'max': vals[-1],
            'mean': mean,
            'std': std,
            'n': n,
        }


class AnomalyDetector:
    __slots__ = ('z_thresh', 'roc_thresh', 'min_samples',
                  'mean', 'm2', 'n', 'prev_val', 'prev_t', 'z_hits', 'roc_hits')
    
    def __init__(self, z_threshold: float = 3.0,
                 roc_threshold: float = 3.0,
                 min_samples: int = 20):
        self.z_thresh = z_threshold
        self.roc_thresh = roc_threshold
        self.min_samples = min_samples
        self.mean = float('nan')
        self.m2 = 0.0
        self.n = 0
        self.prev_val = float('nan')
        self.prev_t = float('nan')
        self.z_hits = 0
        self.roc_hits = 0
    
    def update(self, rec: TelemetryRecord) -> Optional[Dict[str, Any]]:
        x = rec.value
        t = rec.timestamp
        anomalies = []
        
        if self.n >= self.min_samples:
            std = math.sqrt(self.m2 / self.n)
            if std > 1e-12:
                z = (x - self.mean) / std
                if abs(z) > self.z_thresh:
                    anomalies.append({'type': 'zscore', 'z': z, 'value': x})
                    self.z_hits += 1
            
            if not math.isnan(self.prev_val) and not math.isnan(self.prev_t):