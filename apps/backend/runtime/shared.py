"""Shared globals and helper holder for backend runtime."""


class VariableHolder:
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)


global_variables = VariableHolder()

__all__ = ["VariableHolder", "global_variables"]
