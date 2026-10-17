"""Low-poly meshes and batched transforms for the what-if scenes.

Every mesh is a flat float32 array of triangle corners (no normals: the shader takes
face normals from screen-space derivatives, which gives the faceted look for free).
Transforms work on whole batches at once, so a crowd of 150 people is a few numpy calls.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# Mesh builders ---------------------------------------------------------------


def _tris(points: list) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def box() -> np.ndarray:
    """Unit cube standing on the origin: x and z in [-0.5, 0.5], y in [0, 1]."""
    v = [(x, y, z) for x in (-0.5, 0.5) for y in (0.0, 1.0) for z in (-0.5, 0.5)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    out = []
    for a, b, c, d in faces:
        out += [v[a], v[b], v[c], v[a], v[c], v[d]]
    return _tris(out)


def limb() -> np.ndarray:
    """Unit cube hanging from the origin (y in [-1, 0]), so it swings from its top."""
    m = box()
    m[:, 1] -= 1.0
    return m


def cylinder(sides: int = 8) -> np.ndarray:
    out = []
    ring = [(0.5 * math.cos(2 * math.pi * i / sides), 0.5 * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    for i in range(sides):
        (x0, z0), (x1, z1) = ring[i], ring[(i + 1) % sides]
        out += [(x0, 0, z0), (x1, 0, z1), (x1, 1, z1), (x0, 0, z0), (x1, 1, z1), (x0, 1, z0)]
        out += [(0, 1, 0), (x0, 1, z0), (x1, 1, z1), (0, 0, 0), (x1, 0, z1), (x0, 0, z0)]
    return _tris(out)


def cone(sides: int = 8) -> np.ndarray:
    out = []
    ring = [(0.5 * math.cos(2 * math.pi * i / sides), 0.5 * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    for i in range(sides):
        (x0, z0), (x1, z1) = ring[i], ring[(i + 1) % sides]
        out += [(x0, 0, z0), (x1, 0, z1), (0, 1, 0), (0, 0, 0), (x1, 0, z1), (x0, 0, z0)]
    return _tris(out)


def icosphere(subdivisions: int = 1) -> np.ndarray:
    """Sphere of radius 0.5 around the origin."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    tris = np.array([[verts[i] for i in f] for f in faces], dtype=np.float64)
    tris /= np.linalg.norm(tris, axis=2, keepdims=True)
    for _ in range(subdivisions):
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        ab, bc, ca = (a + b), (b + c), (c + a)
        ab, bc, ca = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (ab, bc, ca))
        tris = np.concatenate([np.stack(f, 1) for f in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    return (tris.reshape(-1, 3) * 0.5).astype(np.float32)


def hull() -> np.ndarray:
    """Boat hull, bow pointing to -z, deck at y=1, keel at y=0, 1x1 footprint."""
    top = [(-0.5, 1, 0.5), (0.5, 1, 0.5), (0.5, 1, -0.15), (0.0, 1, -0.5), (-0.5, 1, -0.15)]
    bottom = [(-0.32, 0, 0.45), (0.32, 0, 0.45), (0.3, 0, -0.1), (0.0, 0.35, -0.48), (-0.3, 0, -0.1)]
    out = []
    n = len(top)
    for i in range(n):
        j = (i + 1) % n
        out += [top[i], top[j], bottom[j], top[i], bottom[j], bottom[i]]
    for i in range(1, n - 1):
        out += [top[0], top[i], top[i + 1], bottom[0], bottom[i + 1], bottom[i]]
    return _tris(out)


def prism() -> np.ndarray:
    """Gable roof: a triangular prism, base on y=0 (x and z in [-0.5, 0.5]), ridge along z at y=1."""
    a, b, c = (-0.5, 0, -0.5), (0.5, 0, -0.5), (0, 1, -0.5)
    d, e, f = (-0.5, 0, 0.5), (0.5, 0, 0.5), (0, 1, 0.5)
    return _tris([a, b, c, d, f, e, a, c, f, a, f, d, b, e, f, b, f, c, a, d, e, a, e, b])


def quad() -> np.ndarray:
    """Flat square on y=0, x and z in [-0.5, 0.5]."""
    return _tris([(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)])


PROTOTYPES = {
    "box": box,
    "limb": limb,
    "cyl": lambda: cylinder(8),
    "cyl5": lambda: cylinder(5),
    "cone": lambda: cone(8),
    "cone5": lambda: cone(5),
    "cone6": lambda: cone(6),
    "ico": lambda: icosphere(1),
    "ico0": lambda: icosphere(0),
    "hull": hull,
    "quad": quad,
    "prism": prism,
}

# Batched transforms ----------------------------------------------------------


def _eye(shape: tuple) -> np.ndarray:
    m = np.zeros(shape + (4, 4))
    m[..., 0, 0] = m[..., 1, 1] = m[..., 2, 2] = m[..., 3, 3] = 1.0
    return m


def tr(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = _eye(p.shape[:-1])
    m[..., :3, 3] = p
    return m


def sc(s) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if s.ndim == 0 or s.shape[-1] != 3:
        s = np.stack([s, s, s], -1)
    m = _eye(s.shape[:-1])
    m[..., 0, 0], m[..., 1, 1], m[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    return m


def rx(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    m = _eye(a.shape)
    m[..., 1, 1], m[..., 1, 2], m[..., 2, 1], m[..., 2, 2] = c, -s, s, c
    return m


def ry(a) -> np.ndarray:
    """Rotation about y; maps local +z (forward) to (sin a, 0, cos a)."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    m = _eye(a.shape)
    m[..., 0, 0], m[..., 0, 2], m[..., 2, 0], m[..., 2, 2] = c, s, -s, c
    return m


def rz(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a), np.sin(a)
    m = _eye(a.shape)
    m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = c, -s, s, c
    return m


def chain(*mats) -> np.ndarray:
    out = mats[0]
    for m in mats[1:]:
        out = np.matmul(out, m)
    return out


def align_y(direction) -> np.ndarray:
    """Rotations taking local +y onto each (unit) direction."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = d @ y
    k = 1.0 / np.maximum(1.0 + c, 1e-6)
    m = _eye(d.shape[:-1])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    m[..., 0, 0] = 1 - (vy * vy + vz * vz) * k
    m[..., 0, 1] = -vz + vx * vy * k
    m[..., 0, 2] = vy + vx * vz * k
    m[..., 1, 0] = vz + vx * vy * k
    m[..., 1, 1] = 1 - (vx * vx + vz * vz) * k
    m[..., 1, 2] = -vx + vy * vz * k
    m[..., 2, 0] = -vy + vx * vz * k
    m[..., 2, 1] = vx + vy * vz * k
    m[..., 2, 2] = 1 - (vx * vx + vy * vy) * k
    return m


def beam(a, b, width: float) -> np.ndarray:
    """A box (the "box" mesh) stretched from point a to point b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    length = np.linalg.norm(d, axis=-1)
    w = np.broadcast_to(np.asarray(width, dtype=float), length.shape)
    return chain(tr(a), align_y(d), sc(np.stack([w, length, w], -1)))


# Colors ----------------------------------------------------------------------


def rgb(hex_color: str, scale: float = 1.0) -> tuple[float, float, float]:
    """'#rrggbb' to linear RGB (the shader lights and tone-maps in linear space)."""
    h = hex_color.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return tuple(c * scale for c in lin)


# Material ids, shared with the fragment shader.
SOLID, WATER, GLOW, BUILDING, STRIPES, MOUNTAIN, FLAME, GROUND = range(8)


class Instances:
    """Per-mesh lists of instances: a model matrix, a color + emissive, and material params.

    Packed per instance as 24 floats: the matrix column by column, then (r, g, b, emissive),
    then (material, seed, p1, p2).
    """

    def __init__(self):
        self.parts: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, mesh: str, mats, color, emissive=0.0, material: int = SOLID, seed=0.0, p1=0.0, p2=0.0) -> None:
        mats = np.asarray(mats, dtype=float).reshape(-1, 4, 4)
        n = len(mats)
        if n == 0:
            return
        packed = np.empty((n, 24), dtype=np.float32)
        packed[:, :16] = mats.transpose(0, 2, 1).reshape(n, 16)
        packed[:, 16:19] = np.broadcast_to(np.asarray(color, dtype=float).reshape(-1, 3), (n, 3))
        packed[:, 19] = np.broadcast_to(np.asarray(emissive, dtype=float).reshape(-1), (n,))
        packed[:, 20] = material
        packed[:, 21] = np.broadcast_to(np.asarray(seed, dtype=float).reshape(-1), (n,))
        packed[:, 22] = np.broadcast_to(np.asarray(p1, dtype=float).reshape(-1), (n,))
        packed[:, 23] = np.broadcast_to(np.asarray(p2, dtype=float).reshape(-1), (n,))
        self.parts[mesh].append(packed)

    def extend(self, other: "Instances") -> None:
        for mesh, chunks in other.parts.items():
            self.parts[mesh].extend(chunks)

    def arrays(self) -> dict[str, np.ndarray]:
        return {mesh: np.concatenate(chunks) for mesh, chunks in self.parts.items() if chunks}
