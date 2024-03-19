from proxyllm.utils.sorting import MinHeap


def test_min_heap_sorts_items_in_order():
    # Arrange
    items = [
        {"name": 2, "value": 0.1, "data": []},
        {"name": 0, "value": 0.0005255, "data": []},
        {"name": 3, "value": 1.0, "data": []},
        {"name": 1, "value": 0.001, "data": []},
    ]

    min_heap = MinHeap()
    for item in items:
        min_heap.push(item["value"], item)

    # Act + Assert
    for i in range(len(items)):
        # ensure that the values min_values popped are from minimum to maximum
        value = min_heap.pop_min()
        assert value.get("data", {}).get("name") == i
