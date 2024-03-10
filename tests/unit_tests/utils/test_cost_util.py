from proxyllm.utils.cost import calculate_estimated_max_cost


def test_calculate_estimated_max_cost():
    # Arrange
    price_data = {"prompt": 0.02, "completion": 0.03}

    # Act
    estimate = calculate_estimated_max_cost(price_data, 100, 300)

    # Assert
    assert estimate == 11.0
