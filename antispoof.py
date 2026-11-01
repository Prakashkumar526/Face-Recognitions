"""Movement-based anti-spoofing (report section 3.1 / 5.1).

A recognised face is accepted as live only if its position changed by more
than SPOOF_THRESHOLD pixels relative to an earlier frame in a short window.
A perfectly still printed photo or screen therefore never gets verified.
"""
import math
import time
from collections import defaultdict, deque

import config


class MovementVerifier:
    def __init__(self, threshold=config.SPOOF_THRESHOLD, window_sec=config.SPOOF_WINDOW_SEC):
        self.threshold = threshold
        self.window_sec = window_sec
        self._tracks = defaultdict(deque)   # label -> deque[(t, (cx, cy))]

    @staticmethod
    def centre(box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, label, box, now=None):
        """Record the face position; return True if movement proves liveness."""
        now = time.monotonic() if now is None else now
        c = self.centre(box)
        track = self._tracks[label]
        while track and now - track[0][0] > self.window_sec:
            track.popleft()
        moved = any(math.dist(c, p) >= self.threshold for _, p in track)
        track.append((now, c))
        return moved

    def reset(self, label=None):
        if label is None:
            self._tracks.clear()
        else:
            self._tracks.pop(label, None)
