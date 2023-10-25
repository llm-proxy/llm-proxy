from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    @classmethod
    def list_values(cls):
        return set(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return set(map(lambda c: c.name, cls))

    @classmethod
    def list_enums(cls):
        return set(map(lambda c: c, cls))
