from proxyllm.utils.enums import BaseEnum


def test_list_values_with_str_values() -> None:
    # Arrange
    class TestEnum(str, BaseEnum):
        JUNIOR_DEV = "E-1"
        INTERMEDIATE_DEV = "E-2"
        SENIOR_DEV = "E-3"

    test_values = {"E-1", "E-2", "E-3"}

    # Act
    values = TestEnum.list_values()

    # Assert
    assert set(values) == test_values


def test_list_name_with_int_values() -> None:
    # Arrange
    class TestEnum(int, BaseEnum):
        MONDAY = 1
        TUESDAY = 2
        WEDNESDAY = 3
        THURSDAY = 4
        FRIDAY = 5
        SATURDAY = 6
        SUNDAY = 7

    test_values = {
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    }

    # Act
    names = TestEnum.list_names()

    # Assert
    assert set(names) == test_values


def test_list_enum_with_float_values() -> None:
    # Arrange
    class TestEnum(float, BaseEnum):
        ONE_FIFTY = 1.50
        TWO_NINETY_NINE = 2.99
        EIGHT_THIRTY = 8.30

    test_names = {
        "ONE_FIFTY",
        "TWO_NINETY_NINE",
        "EIGHT_THIRTY",
    }

    test_values = {1.50, 2.99, 8.30}

    # Act
    enums = TestEnum.list_enums()

    # Assert
    assert set(enum.name for enum in enums) == test_names
    assert set(enum.value for enum in enums) == test_values


def test_enum_with_nonexistent_values() -> None:
    # Arrange
    class TestEnum(int, BaseEnum):
        NEG_ONE = -1
        NEG_TWO = -2
        NEG_THREE = -3

    # Act
    test_values = TestEnum.list_values()

    # Assert
    assert 1 not in TestEnum
    assert 2 not in test_values
