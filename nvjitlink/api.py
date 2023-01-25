from enum import Enum


class InputType(Enum):
    NONE = 0
    CUBIN = 1
    PTX = 2
    LTOIR = 3
    FATBIN = 4
    OBJECT = 5
    LIBRARY = 6
