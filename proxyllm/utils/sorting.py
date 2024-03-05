import heapq


class MinHeap:
    """
    A min-heap data structure implementation that uses a heap queue algorithm.

    This class provides a way to maintain a collection of items sorted such that the smallest
    element can be quickly extracted. This implementation ensures that elements are ordered first
    by their value, and then by the order they were added to maintain stability.

    Attributes:
        heap (List[Tuple]): The underlying data structure, a list used to represent the heap.
        counter (int): A counter used to maintain the insertion order of elements.
    """

    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, value, data):
        """
        Adds a new item to the heap.

        Args:
            value (float): The value by which to order the item in the heap.
            data (Any): The data associated with the item.
        """
        entry = (value, self.counter, data)
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def pop_min(self):
        """
        Removes and returns the smallest item from the heap.

        Returns:
            dict: A dictionary containing the 'data' of the smallest item, or None if the heap is empty.
        """
        if self.heap:
            _, _, data = heapq.heappop(self.heap)
            return {"data": data}

        return None
