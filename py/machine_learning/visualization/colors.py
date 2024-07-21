
MAX_COLOR_NUMBER = 10

COLOR_PALETTE_10 = (
    "#3288bd",
    "#9e0142",
    "#66c2a5",
    "#f46d43",
    "#5e4fa2",
    "#d53e4f",
    "#fdae61",
    "#abdda4",
    "#fee08b",
    "#e6f598",

)


def get_color_palette(n):
    if n <= 10:
        return COLOR_PALETTE_10[:n]
    else:
        raise RuntimeError(f"Cannot support more than {n} colors")
