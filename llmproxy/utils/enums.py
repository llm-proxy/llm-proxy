from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """
    A metaclass for Enums that enables direct containment checks for enum values.

    This metaclass enhances standard Enum functionality by allowing one to check
    if a given item is a valid value of the enum associated with this metaclass.
    """

    def __contains__(cls, item):
        """
        Check if the enum contains a given item.

        Args:
            item: The item to check for containment within the enum values.

        Returns:
            bool: True if the item is a valid value of the enum; False otherwise.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    """
    Base class for enums used within the application.

    Provides additional methods to list the values, names, or enum instances.
    Inherits from Enum and uses MetaEnum to allow for enhanced functionality.
    """

    @classmethod
    def list_values(cls):
        """
        List all the values of the enum.

        Returns:
            set: A set containing all the values of the enum.
        """
        return set(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        """
        List all the names of the enum members.

        Returns:
            set: A set containing all the names of the enum members.
        """
        return set(map(lambda c: c.name, cls))

    @classmethod
    def list_enums(cls):
        """
        List all the enum members.

        Returns:
            set: A set containing all the enum members.
        """
        return set(map(lambda c: c, cls))
