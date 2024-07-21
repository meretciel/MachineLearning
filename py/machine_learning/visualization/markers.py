
POINT = "."
CIRCLE = "o"
TRIANGLE_DOWN = "v"
TRIANGLE_UP = "^"
TRIANGLE_LEFT = "<"
TRIANGLE_RIGHT = ">"
SQUARE = "s"
PLUS = "+"
CROSS = "x"
DIAMOND = "D"


MARKER_LIST = (
    "o", "s", "^", "v", "<", ">", "x", "+", "p", "H"
)


def get_marker_list(n):
    assert n < 10
    return MARKER_LIST
