"""Pixel-art nhỏ (vẽ bằng ô vuông trên Canvas): ô "?", xu, sao, nấm, Goomba."""
import tkinter as tk

from app import theme

PAL = {
    "K": theme.BLACK, "Y": theme.COIN, "D": theme.COIN_DARK, "W": "#FFF0B0",
    "R": theme.RED, "H": "#FFFFFF", "B": "#A85800", "T": "#F8D8A0", "G": theme.PIPE,
    "L": theme.PIPE_LIGHT, "S": theme.SKY, "N": theme.BRICK, "M": theme.BRICK_DARK,
}

SPRITES = {
    "qblock": [
        "KKKKKKKKKKKKKK",
        "KWYYYYYYYYYYDK",
        "KYYYYDDDDYYYDK",
        "KYYYDDYYDDYYDK",
        "KYYYYYYYDDYYDK",
        "KYYYYYYDDYYYDK",
        "KYYYYYDDYYYYDK",
        "KYYYYYDDYYYYDK",
        "KYYYYYYYYYYYDK",
        "KYYYYYDDYYYYDK",
        "KYYYYYDDYYYYDK",
        "KYYYYYYYYYYYDK",
        "KDDDDDDDDDDDDK",
        "KKKKKKKKKKKKKK",
    ],
    "coin": [
        "...KKKK...",
        "..KYYYYK..",
        ".KYYWYYDK.",
        ".KYYWYYDK.",
        "KYYWYYYYDK",
        "KYYWYYYYDK",
        "KYYWYYYYDK",
        "KYYWYYYYDK",
        "KYYWYYYYDK",
        "KYYWYYYYDK",
        ".KYYWYYDK.",
        ".KYYWYYDK.",
        "..KYYYYK..",
        "...KKKK...",
    ],
    "star": [
        ".......KK.....",
        "......KYYK....",
        "......KYYK....",
        ".....KYYYYK...",
        "KKKKKKYYYYKKKK",
        "KYYYYYYYYYYYYK",
        ".KYYYYYYYYYYK.",
        "..KYYKYYKYYK..",
        "..KYYKYYKYYK..",
        "..KYYYYYYYYK..",
        ".KYYYYKKYYYYK.",
        ".KYYYK..KYYYK.",
        "KYYYK....KYYYK",
        "KKKK......KKKK",
    ],
    "mushroom": [
        "....KKKKKK....",
        "..KKRRHHRRKK..",
        ".KRRHHHHRRRRK.",
        ".KRHHHHRRHHRK.",
        "KRRHHHHRRHHRRK",
        "KRRRHHRRRRRRRK",
        "KRRRRRRRRRRRRK",
        "KKKKKKKKKKKKKK",
        ".KTTKTTTTKTTK.",
        ".KTTKTTTTKTTK.",
        ".KTTTTTTTTTTK.",
        "..KTTTTTTTTK..",
        "...KKKKKKKK...",
    ],
    "goomba": [
        "......KKKK......",
        "....KKBBBBKK....",
        "...KBBBBBBBBK...",
        "..KBBBBBBBBBBK..",
        ".KBBHHBBBBHHBBK.",
        ".KBBHKHBBHKHBBK.",
        "KBBBHKHBBHKHBBBK",
        "KBBBHHHBBHHHBBBK",
        "KBBBBBBBBBBBBBBK",
        ".KKKKTTTTTTKKKK.",
        "...KTTTTTTTTK...",
        "...KTTTTTTTTK...",
        "....KKTTTTKK....",
        "...KKKKKKKKKK...",
        "..KKKKK..KKKKK..",
        "..KKKK....KKKK..",
    ],
}


def size_of(name, scale=2):
    rows = SPRITES[name]
    return len(rows[0]) * scale, len(rows) * scale


def draw_sprite(canvas, name, x=0, y=0, scale=2, tag="sprite"):
    for r, row in enumerate(SPRITES[name]):
        for c, ch in enumerate(row):
            color = PAL.get(ch)
            if color:
                canvas.create_rectangle(x + c * scale, y + r * scale, x + (c + 1) * scale,
                                        y + (r + 1) * scale, fill=color, outline=color, tags=tag)


class Sprite(tk.Canvas):
    """Canvas nhỏ chứa 1 sprite, đổi sprite bằng .set()."""

    def __init__(self, parent, name="qblock", scale=2, bg=theme.PANEL):
        w, h = 34 * 1, 34
        super().__init__(parent, width=w, height=h, bg=bg, highlightthickness=0, bd=0)
        self.scale = scale
        self.name = None
        self.set(name)

    def set(self, name):
        if name == self.name:
            return
        self.name = name
        self.delete("all")
        w, h = size_of(name, self.scale)
        self.configure(width=w + 2, height=h + 2)
        draw_sprite(self, name, 1, 1, self.scale)
