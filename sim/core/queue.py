import heapq

class EventQueue:
    def __init__(self):
        self._h = []

    def push(self, e):
        heapq.heappush(self._h, (e.ts, e.seq, e))

    def pop(self):
        return heapq.heappop(self._h)[2]

    def __len__(self):
        return len(self._h)