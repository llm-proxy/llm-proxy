import heapq


class MinHeap:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, value, data):
        entry = (value, self.counter, data)
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def pop_min(self):
        if self.heap:
            _, _, data = heapq.heappop(self.heap)
            return {"data": data}
        else:
            return None
