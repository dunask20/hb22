#!/usr/bin/env python3
"""
gift.py — вращающийся подарок с бантом, нарисованный буквами.

Как это работает:
  1. Подарок собирается из облака 3D-точек (коробка, крышка, ленты, бант, узел).
     У каждой точки есть положение, нормаль, цвет и «блеск».
  2. Каждый кадр облако поворачивается, освещается и проецируется на экран.
  3. Экран разбит на сетку символов. В каждой ячейке остаётся ближайшая точка,
     а символ выбирается по яркости: . : + * # @
  4. Кадры собираются в mp4 через ffmpeg.

Запуск:
    python3 gift.py                 # gift.mp4 (720x1280, 12 c) + gift_still.png
    python3 gift.py --still         # только картинка
    python3 gift.py --seconds 20 --out my.mp4

Нужно: numpy, Pillow, ffmpeg.
"""
import argparse
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- настройки
W, H = 720, 1280              # размер видео
CW, CH = 8, 13                # размер одной «буквы» в пикселях
COLS, ROWS = W // CW, H // CH
FPS = 30
RHO = 3600                    # плотность точек на единицу площади
PITCH = np.radians(22)        # наклон камеры сверху
SCALE = 195                   # пикселей на единицу длины
CAM_D = 9.0                   # расстояние до камеры (перспектива)
SPIN = 0.55                   # скорость вращения, рад/с
RAMP = " .,:;-=+*xX#%@"       # от тёмного к яркому
STAR, PLUS, DOT = RAMP.index("*"), RAMP.index("+"), RAMP.index(".")

# палитра (0..1)
PAPER = np.array([0.80, 0.07, 0.13])    # красная упаковка
DOTC = np.array([1.00, 0.93, 0.85])     # белые горошины
GOLD = np.array([1.00, 0.74, 0.22])     # золотая лента
CHAMP = np.array([1.00, 0.88, 0.58])    # шампань: внутренние петли банта
SNOW = np.array([0.62, 0.70, 0.90])     # снег на земле

# появление частей подарка: группа -> (начало, длительность) в секундах
REVEAL = {0: (0.0, 1.0), 1: (0.5, 1.8), 2: (2.0, 1.2),
          3: (3.0, 1.2), 4: (4.0, 1.8), 5: (5.0, 1.2)}
G_GROUND, G_BODY, G_LID, G_RIBBON, G_BOW, G_KNOT = range(6)

rng = np.random.default_rng(7)
UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- геометрия
class Cloud:
    def __init__(self):
        self.P, self.N, self.C, self.S, self.G, self.T = [], [], [], [], [], []

    def add(self, p, n, c, shine, group, two_sided=False):
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        m = len(p)
        self.P.append(p)
        self.N.append(n)
        self.C.append(c)
        self.S.append(np.full(m, shine))
        self.G.append(np.full(m, group))
        self.T.append(np.full(m, two_sided))

    def done(self):
        return [np.concatenate(v) for v in (self.P, self.N, self.C, self.S, self.G, self.T)]


def rect(c, u, v):
    """Случайные точки на прямоугольнике: центр c, полу-векторы u и v."""
    m = int(RHO * 4 * np.linalg.norm(u) * np.linalg.norm(v))
    a, b = rng.uniform(-1, 1, m), rng.uniform(-1, 1, m)
    return c + a[:, None] * u + b[:, None] * v


def paper(s, t):
    """Красная бумага с белыми горошинами в шахматном порядке."""
    fx = s * 2.4
    fy = t * 2.4 + 0.5 * (np.floor(fx) % 2)
    dot = ((fx % 1) - .5) ** 2 + ((fy % 1) - .5) ** 2 < 0.028
    col = np.tile(PAPER, (len(s), 1)) * (0.92 + 0.16 * rng.random((len(s), 1)))
    col *= (1 + 0.04 * np.sin(s * 9 + t * 5))[:, None]       # лёгкие складки
    col[dot] = DOTC
    return col


def occlusion(y):
    """Тень под крышкой и у земли."""
    return (1 - 0.55 * np.clip((y - 1.22) / 0.18, 0, 1)) * (1 - 0.25 * np.clip((0.25 - y) / 0.25, 0, 1))


def strip(C, w, wid):
    """Лента: точки вокруг кривой C (K,3), w — направление ширины, wid — полуширина."""
    K = len(C)
    T = np.gradient(C, axis=0)
    seg = np.linalg.norm(T, axis=1)
    m = int(RHO * seg.sum() * 2 * wid)
    ti = rng.uniform(0, K - 1.001, m)
    i0 = ti.astype(int)
    f = (ti - i0)[:, None]
    cp = C[i0] * (1 - f) + C[i0 + 1] * f
    wv = np.broadcast_to(w, C.shape)[i0]
    r = rng.uniform(-1, 1, m)
    p = cp + wv * wid * r[:, None]
    n = np.cross(T[i0] / (seg[i0][:, None] + 1e-9), wv)
    return p, n, ti / (K - 1), r


def build():
    cl = Cloud()
    e = np.eye(3)

    # --- земля (снег)
    m = 60000
    rad = 3.6 * np.sqrt(rng.random(m))
    ang = rng.uniform(0, 2 * np.pi, m)
    p = np.stack([rad * np.cos(ang), np.zeros(m), rad * np.sin(ang)], 1)
    fall = np.exp(-(rad / 2.9) ** 2.5)
    col = SNOW[None] * (0.32 * fall * (0.6 + 0.4 * rng.random(m)))[:, None]
    cl.add(p, np.tile(UP, (m, 1)), col, 0.0, G_GROUND)

    # --- коробка и крышка: боковые грани
    for grp, hw, yc, hh, shine in ((G_BODY, 1.0, 0.7, 0.7, 0.15), (G_LID, 1.07, 1.575, 0.175, 0.25)):
        for ax in (0, 2):
            for sg in (-1, 1):
                n = np.zeros(3); n[ax] = sg
                ot = 2 if ax == 0 else 0
                c = n * hw + np.array([0, yc, 0])
                p = rect(c, e[ot] * hw, UP * hh)
                col = paper(p[:, ot] + (ax * 3 + sg) * 0.37, p[:, 1])
                col *= occlusion(p[:, 1])[:, None]
                cl.add(p, np.tile(n, (len(p), 1)), col, shine, grp)
    # верх крышки
    hw = 1.07
    p = rect(np.array([0, 1.75, 0]), e[0] * hw, e[2] * hw)
    cl.add(p, np.tile(UP, (len(p), 1)), paper(p[:, 0] + 1.1, p[:, 2]), 0.25, G_LID)

    # --- золотые ленты: крест вокруг коробки
    def gold(p):
        return np.tile(GOLD, (len(p), 1)) * (0.93 + 0.07 * np.sin(p[:, 1] * 80 + p[:, 0] * 40))[:, None]

    for ax in (0, 2):
        for sg in (-1, 1):
            n = np.zeros(3); n[ax] = sg
            ot = 2 if ax == 0 else 0
            for off, yc, hh in ((1.015, 0.7, 0.7), (1.085, 1.575, 0.175)):
                p = rect(n * off + np.array([0, yc, 0]), e[ot] * 0.17, UP * hh)
                col = gold(p) * (occlusion(p[:, 1]) if off < 1.05 else 1)[:, None] if off < 1.05 else gold(p)
                cl.add(p, np.tile(n, (len(p), 1)), col, 0.85, G_RIBBON)
    for u, v in ((e[0] * hw, e[2] * 0.17), (e[0] * 0.17, e[2] * hw)):
        p = rect(np.array([0, 1.765, 0]), u, v)
        cl.add(p, np.tile(UP, (len(p), 1)), gold(p), 0.85, G_RIBBON)

    # --- бант
    base = np.array([0.0, 1.77, 0.0])

    def loop(phi, A, B, lift, h0, wid, color):
        th = np.linspace(0, 2 * np.pi, 400)
        s = A * (1 - np.cos(th)) / 2                      # 0 -> A -> 0
        hh = B * np.sin(th) * (0.55 + 0.45 * s / A)      # каплевидная петля
        h = h0 + s * lift + hh
        d = np.array([np.cos(phi), 0, np.sin(phi)])
        w = np.array([-np.sin(phi), 0, np.cos(phi)])
        C = base + d * s[:, None] + UP * h[:, None]
        p, n, u, r = strip(C, w, wid)
        sp = A * (1 - np.cos(np.interp(u, np.linspace(0, 1, 400), th))) / 2
        col = np.tile(color, (len(p), 1)) * (0.72 + 0.32 * sp / A + 0.06 * r)[:, None]
        cl.add(p, n, col, 0.9, G_BOW, True)

    for k in range(6):
        loop(k * np.pi / 3 + 0.3, 0.95 if k % 2 == 0 else 0.78, 0.24, 0.75, 0.05, 0.12, GOLD)
    for k in range(4):
        loop(k * np.pi / 2 + np.pi / 4 + 0.3, 0.55, 0.16, 1.05, 0.10, 0.10, CHAMP)

    # хвосты банта (с «ласточкиным» вырезом)
    for phi, wave in ((np.radians(35), 0.0), (np.radians(215), 1.7)):
        Lt = 1.38
        u = np.linspace(0.05, Lt, 200)
        d = np.array([np.cos(phi), 0, np.sin(phi)])
        w = np.array([-np.sin(phi), 0, np.cos(phi)])
        C = (base + UP * 0.012) + d * u[:, None] + w * (0.16 * np.sin(u * 2.6 + wave))[:, None] \
            + UP * (0.05 * (u / Lt) ** 2)[:, None]
        p, n, uu, r = strip(C, w, 0.17)
        q = np.clip((uu - 0.9) / 0.1, 0, 1)
        keep = q <= np.abs(r)
        p, n, uu = p[keep], n[keep], uu[keep]
        col = np.tile(GOLD, (len(p), 1)) * (0.8 + 0.25 * uu)[:, None]
        cl.add(p, n, col, 0.9, G_KNOT, True)

    # узел банта
    m = int(RHO * 3)
    v = rng.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad3 = np.array([0.30, 0.22, 0.30])
    p = base + np.array([0, 0.10, 0]) + v * rad3
    cl.add(p, v / rad3, np.tile(CHAMP, (m, 1)) * 0.95, 0.95, G_KNOT, True)

    P, N, C, S, G, T = cl.done()
    P = P - np.array([0, 1.15, 0])                        # центрируем по высоте
    return (P.astype(np.float32), N.astype(np.float32), C.astype(np.float32),
            S.astype(np.float32), G, T)


# ---------------------------------------------------------------- рендер
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def load_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "DejaVuSansMono-Bold.ttf", "Menlo.ttc", "Consolas.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_atlas():
    font = load_font(13)
    atlas = np.zeros((len(RAMP), CH, CW), np.float32)
    for i, ch in enumerate(RAMP):
        if ch == " ":
            continue
        im = Image.new("L", (CW, CH), 0)
        ImageDraw.Draw(im).text((0, CH - 3), ch, font=font, fill=255, anchor="ls")
        atlas[i] = np.asarray(im, np.float32) / 255
    return atlas


def make_background():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = (1 - y) * np.array([5, 7, 18], np.float32) + y * np.array([14, 17, 40], np.float32)
    bg = np.repeat(bg, W, axis=1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-(((xx - W / 2) / 330) ** 2 + ((yy - H * 0.52) / 380) ** 2))
    return bg + glow[..., None] * np.array([26, 18, 30], np.float32)


class Scene:
    def __init__(self):
        self.P, self.N, self.C, self.S, self.G, self.T = build()
        self.n = len(self.P)
        self.key = rng.random(self.n).astype(np.float32)       # для эффекта «проявления»
        self.ground = self.G == G_GROUND
        self.gxz = self.P[self.ground]
        self.sparkle_mask = np.isin(self.G, (G_RIBBON, G_BOW, G_KNOT))
        self.atlas = make_atlas()
        self.bg = make_background()
        self.font = load_font(11)
        nf = 220                                                # снежинки
        self.fl = dict(x=rng.uniform(-4.2, 4.2, nf), z=rng.uniform(-3.5, 3.5, nf),
                       y=rng.uniform(0, 8, nf), v=rng.uniform(0.5, 1.2, nf),
                       ph=rng.uniform(0, 6.28, nf), b=rng.uniform(0.6, 1.0, nf),
                       ch=rng.choice([DOT, STAR, PLUS, DOT], nf))
        self.L = np.array([-0.35, 0.55, 0.80], np.float32); self.L /= np.linalg.norm(self.L)
        self.Hh = self.L + np.array([0, 0, 1], np.float32); self.Hh /= np.linalg.norm(self.Hh)

    def render(self, t, frame_no=0):
        a = t * SPIN
        R, Rp = rot_y(a), rot_x(PITCH)
        # поворот подарка (земля не вращается)
        Pw, Nw = self.P @ R.T, self.N @ R.T
        Pw[self.ground], Nw[self.ground] = self.P[self.ground], self.N[self.ground]
        V, NV = Pw @ Rp.T, Nw @ Rp.T

        # освещение
        flip = self.T & (NV[:, 2] < 0)
        NV[flip] *= -1
        diff = np.clip(NV @ self.L, 0, None)
        fill = np.clip(NV @ np.array([0.8, 0.1, 0.3], np.float32), 0, None)
        lit = self.C * (0.42 + 0.95 * diff)[:, None] + 0.15 * fill[:, None] * np.array([0.4, 0.5, 1.0], np.float32)
        spec = self.S * np.clip(NV @ self.Hh, 0, None) ** 28
        lit += (spec * 1.2)[:, None] * np.array([1, .95, .85], np.float32)
        rim = (1 - np.clip(NV[:, 2], 0, 1)) ** 3 * 0.22
        lit += rim[:, None] * np.array([0.25, 0.4, 1.0], np.float32) * (~self.ground)[:, None]
        # мягкая тень подарка на снегу
        g = self.gxz @ R
        d = np.maximum(np.abs(g[:, 0]), np.abs(g[:, 2])) - 1.0
        lit[self.ground] *= (1 - 0.85 * np.exp(-np.clip(d, 0, None) * 1.8))[:, None]
        # искры на золоте
        sp = np.random.default_rng(frame_no // 3).random(self.n) < 0.0025
        sp &= self.sparkle_mask
        lit[sp] += 1.2
        lum = np.clip(lit @ np.array([0.30, 0.59, 0.11], np.float32), 0, 1)
        chidx = np.clip((lum ** 0.85 * (len(RAMP) - 1)).astype(int) + (lum > 0.03), 0, len(RAMP) - 1)
        chidx[sp] = STAR

        # проявление частей по времени
        prog = np.zeros(len(REVEAL), np.float32)
        for gi, (t0, du) in REVEAL.items():
            prog[gi] = np.clip((t - t0) / du, 0, 1)
        vis = self.key < prog[self.G]

        # снег
        f = self.fl
        fy = (f["y"] - t * f["v"]) % 8 - 3.6
        fx = f["x"] + 0.12 * np.sin(t * 1.3 + f["ph"])
        FV = np.stack([fx, fy, f["z"]], 1).astype(np.float32)
        fcol = np.outer(f["b"], [0.88, 0.94, 1.0]).astype(np.float32)

        V = np.concatenate([V[vis], FV])
        col = np.concatenate([lit[vis], fcol])
        ch = np.concatenate([chidx[vis], f["ch"]])

        # проекция и z-буфер по ячейкам
        sc = CAM_D / (CAM_D - V[:, 2])
        cc = ((W / 2 + V[:, 0] * SCALE * sc) // CW).astype(int)
        rr = ((H * 0.50 - V[:, 1] * SCALE * sc) // CH).astype(int)
        ok = (cc >= 0) & (cc < COLS) & (rr >= 0) & (rr < ROWS)
        order = np.argsort(V[ok, 2])                           # дальние -> ближние
        fl = (rr * COLS + cc)[ok][order]
        cell_c = np.zeros((ROWS * COLS, 3), np.float32)
        cell_h = np.zeros(ROWS * COLS, int)
        cell_c[fl] = col[ok][order]
        cell_h[fl] = ch[ok][order]

        # сборка картинки из атласа символов
        gl = self.atlas[cell_h.reshape(ROWS, COLS)]                    # (R,C,CH,CW)
        img = gl[..., None] * (np.clip(cell_c * 1.5, 0, 1.4) * 255).reshape(ROWS, COLS, 1, 1, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(ROWS * CH, COLS * CW, 3)
        out = self.bg.copy()
        out[:ROWS * CH, :COLS * CW] += img
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

        dr = ImageDraw.Draw(im)
        dr.text((W - 14, 12), f"t {t:05.2f}s", font=self.font, fill=(255, 90, 120), anchor="ra")
        dr.text((14, H - 62), "@gift.py · 2026", font=self.font, fill=(120, 130, 160))
        dr.text((14, H - 46), f"rotation {np.degrees(a) % 360:06.2f}°   points {int(vis.sum()):,}",
                font=self.font, fill=(120, 130, 160))
        dr.text((14, H - 30), "py gift.py", font=self.font, fill=(120, 130, 160))
        return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gift.mp4")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--still", action="store_true", help="только PNG")
    args = ap.parse_args()

    scene = Scene()
    scene.render(args.seconds * 0.75, 9999).save("gift_still.png")
    print("gift_still.png готов")
    if args.still:
        return

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", args.out], stdin=subprocess.PIPE)
    total = int(args.seconds * FPS)
    for i in range(total):
        ff.stdin.write(scene.render(i / FPS, i).tobytes())
        if i % 30 == 0:
            print(f"кадр {i}/{total}")
    ff.stdin.close()
    ff.wait()
    print(f"{args.out} готов")


if __name__ == "__main__":
    main()
