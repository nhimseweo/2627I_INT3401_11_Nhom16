"""Giao diện phong cách Super Mario (NES): màu, font, style ttk, style matplotlib."""
import os

# Không import tkinter ở đầu file: plugin của worker (train/autograder) cũng import theme
# để lấy màu, và worker không được phụ thuộc vào tkinter.

# ---------------------------------------------------------------- bảng màu (NES)
SKY = "#5C94FC"        # trời
SKY_DARK = "#3C74DC"
BRICK = "#C84C0C"      # gạch
BRICK_DARK = "#7C2C00"
BRICK_LIGHT = "#E4885C"
PANEL = "#FFF3D6"      # nền bảng (kem)
PANEL_DARK = "#F0DCA8"
INK = "#2B1B0E"        # chữ
RED = "#D82800"        # áo Mario
BLUE = "#0058F8"       # quần Mario
PIPE = "#00A800"       # ống cống
PIPE_LIGHT = "#80D010"
PIPE_DARK = "#005800"
COIN = "#FCBC3C"       # xu / ô "?"
COIN_DARK = "#AC7C00"
WHITE = "#FFFFFF"
BLACK = "#000000"
GRAY = "#9C8C7C"

MSG_COLORS = {"info": "#5B4636", "ok": "#007800", "warn": "#B25A00", "error": "#C00000"}

_PIXEL_FONTS = ("Press Start 2P", "Silkscreen", "Pixelify Sans", "VT323")
_FALLBACKS = ("Courier New", "Consolas", "Menlo", "DejaVu Sans Mono", "TkFixedFont")
FAMILY = "TkFixedFont"
PIXEL = False   # True nếu máy có font pixel thật


def pick_font(root):
    """Mặc định dùng Courier New đậm (có đủ dấu tiếng Việt).
    Muốn font pixel thật: cài "Press Start 2P" rồi đặt biến môi trường ML_PIXEL_FONT=1
    (lưu ý font pixel không có dấu tiếng Việt)."""
    global FAMILY, PIXEL
    from tkinter import font as tkfont

    names = set(tkfont.families(root))
    if os.environ.get("ML_PIXEL_FONT") == "1":
        for f in _PIXEL_FONTS:
            if f in names:
                FAMILY, PIXEL = f, True
                return FAMILY
    for f in _FALLBACKS:
        if f in names:
            FAMILY = f
            break
    return FAMILY


def font(size=10, bold=True):
    # font pixel thật rộng hơn nhiều -> giảm cỡ chữ cho vừa khung
    if PIXEL:
        size = max(6, int(size * 0.72))
    return (FAMILY, size, "bold" if bold else "normal")


# ---------------------------------------------------------------- ttk
def apply_theme(root):
    from tkinter import ttk

    pick_font(root)
    root.configure(bg=SKY)
    s = ttk.Style(root)
    s.theme_use("clam")   # clam cho phép đổi màu nút trên cả Windows / macOS / Linux

    s.configure(".", background=PANEL, foreground=INK, font=font(10), bordercolor=BLACK,
                focuscolor=PANEL, troughcolor=BRICK_DARK)
    s.configure("TFrame", background=PANEL)
    s.configure("Sky.TFrame", background=SKY)
    s.configure("TLabel", background=PANEL, foreground=INK, font=font(10))
    s.configure("Sky.TLabel", background=SKY, foreground=WHITE, font=font(10))
    s.configure("Small.TLabel", background=PANEL, foreground=INK, font=font(8))
    s.configure("Hint.TLabel", background=PANEL, foreground="#7A6652", font=font(9, False))
    s.configure("TCheckbutton", background=PANEL, foreground=INK, font=font(10))
    s.map("TCheckbutton", background=[("active", PANEL)])

    s.configure("TEntry", fieldbackground=WHITE, foreground=INK, bordercolor=BLACK,
                lightcolor=BLACK, darkcolor=BLACK, borderwidth=2, padding=3)
    s.configure("TScrollbar", background=BRICK, troughcolor=BRICK_DARK, bordercolor=BLACK,
                arrowcolor=WHITE, lightcolor=BRICK_LIGHT, darkcolor=BRICK_DARK)

    def button(name, bg, fg, active_bg, active_fg, light, dark):
        s.configure(name, background=bg, foreground=fg, font=font(10), padding=(8, 4),
                    borderwidth=3, relief="raised", bordercolor=BLACK, lightcolor=light,
                    darkcolor=dark, focuscolor=bg, anchor="center")
        s.map(name,
              background=[("disabled", GRAY), ("pressed", dark), ("active", active_bg)],
              foreground=[("disabled", "#E8DCC8"), ("active", active_fg)],
              relief=[("pressed", "sunken")],
              lightcolor=[("pressed", dark)], darkcolor=[("pressed", light)])

    button("TButton", RED, WHITE, "#FF5030", WHITE, "#FF7860", "#8C1800")        # đỏ Mario
    button("Pipe.TButton", PIPE, WHITE, PIPE_LIGHT, INK, PIPE_LIGHT, PIPE_DARK)   # ống cống
    button("Coin.TButton", COIN, INK, "#FFD870", INK, "#FFE8A0", COIN_DARK)       # xu vàng
    button("Blue.TButton", BLUE, WHITE, "#3C88FF", WHITE, "#6CA4FF", "#00308C")   # quần xanh
    return s


# ---------------------------------------------------------------- matplotlib
_mpl_done = False


def apply_matplotlib():
    global _mpl_done
    if _mpl_done:
        return
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": PANEL,
        "axes.facecolor": "#FFFBEF",
        "axes.edgecolor": BLACK,
        "axes.linewidth": 1.8,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#E4D3A8",
        "legend.facecolor": "#FFFBEF",
        "legend.edgecolor": BLACK,
        "font.family": "DejaVu Sans",
    })
    _mpl_done = True


def heat_cmap():
    """Bản đồ màu cho heatmap attention: kem → vàng xu → đỏ → nâu gạch."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("mario", ["#FFF3D6", COIN, RED, "#5A1500"])
