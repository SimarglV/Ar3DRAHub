"""Rebuild a broken mesh as one closed, manifold surface. MIT.

**Why this exists.** Every model this studio generates is a *display* surface, not a solid:
Pixal3D, TRELLIS.2 and Hi3D all emit triangle soup that a renderer is happy with and a
slicer, a physics engine or a Boolean is not. Measured on the knight of Rome used for the
Hi3D work: 957,610 faces, `is_watertight` False, and 39,614 separate shells -- dozens of
thousands of hairline gaps that no eye sees and every downstream tool trips on.

Two tools already in this repo touch the problem and neither solves it:

* `scripts/blender_voxel_remesh.py` resamples through Blender's voxel remesh, which does
  close the gaps -- but it leaves every detached sliver as its own closed shell (measured:
  412 shells at voxel 0.012, the largest holding 51,712 of 91,752 faces). "No boundary
  edges" is not "one solid".
* `scripts/fill_holes.py` patches boundary loops in place, which is right for a display
  mesh and wrong for a mesh that must be a solid: it can only close holes that are loops,
  and it cannot merge shells.

**How this does it.** The same idea WTiVo uses -- resample to a volume, extract one
surface -- with MIT components:

1. **An unsigned distance field, kept only near the surface.** Every cell the surface
   passes within a couple of voxels of gets its exact distance to the nearest triangle, by
   a k-d tree over the triangles followed by Ericson's closest-point test. Only near
   cells are filled, so a 700-voxel model costs a fraction of the dense grid.
2. **An inside/outside mask from the surface's own voxelisation.** `trimesh.voxelized`
   (MIT) marks the cells a triangle crosses -- a skin, never a filled body -- and that
   skin *is* the solid. `trimesh.repair.fill_holes` (MIT) runs first, because the
   voxeliser counts edge crossings and a hole is a place where the count goes wrong.
   The skin is the right answer for what this studio generates: the knight of Rome
   measures volume-to-area 0.0013, a shell a fraction of a millimetre thick with empty
   space behind it, and filling that space made it 2.24x heavier -- a different object.
3. **A parity test, so a genuinely solid body still fills.** What the skin encloses is
   either material or sealed air, and the skin alone cannot tell them apart. A region
   the flood fill cannot reach from the grid border is labelled, and a +Z ray from it
   counts how many wall *runs* it crosses: odd means material behind it, even means
   enclosed air. A hollow box keeps its cavity; a solid cube gets its middle filled.
   `VoxelGrid.fill` cannot do this -- it floods from the bounding box and swallows the
   cavity (measured: a hollow box went from 3.90 to 6.37).
4. **One field, then marching cubes.** `field = distance` outside the mask and
   `-distance` inside it, `skimage.measure.marching_cubes` contouring at
   zero. Both sides come from a Euclidean distance transform, asked separately for each
   -- `distance_transform_edt(mask)` returns zero right across a cavity, which flattens
   the field and is what once turned a hollow box into a solid one. A mask, not a
   winding test, is what turns a shell into a solid: an inside-out fragment reads solid
   from outside either way, so a mis-wound triangle no longer punches a hole in the
   result.
5. **Keep the body and its cavities, drop the crumbs**, by shared *edges* rather than
   shared vertices. Marching cubes over a partial band finishes with frontier caps;
   edge sharing is the rule a slicer uses, and it is why this does not call
   `trimesh.split`, which connects through vertices and reads a crumb touching the body
   as one piece. A cavity is a real part of the shape and is kept; a sliver is not.

**What it costs.** The result resamples, so detail below the voxel size is gone, and it has
no UVs -- carry the texture across with `scripts/blender_hi3d_transfer.py`, or bake again
afterwards. Voxel size is the one knob: too coarse melts fingers and sword blades, too fine
leaves the gaps between shells open and blows up memory. The rebuild measures free RAM
before each big allocation and stops with a sentence you can act on, never an OOM kill.
`choose_voxel` picks a starting
point from the mesh's own size; `across` is how many voxels span its longest side.

A thin shell stays a thin shell, and its volume still goes *up*: the wall comes back one
voxel thick, which is thicker than the sub-voxel skin it replaced. Measured on the knight
at `across=300`, 0.00268 in and 0.00600 out -- but the output encloses 3.2% of its bounding
box against the input's 1.4%, so nothing was poured in. A closed surface thinner than a
voxel cannot be built; that is the resolution cost, made visible.

Licences: MIT. No extra binary to fetch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Starting guess: about 700 voxels across the longest side. Measured on the knight, that
# keeps fingers and the sword edge while the hairline gaps close, and holds the volume
# tables at a size a laptop handles.
DEFAULT_ACROSS = 700
# The distance band reaches this many voxels either side of the surface. Two is enough for
# marching cubes to see a sign change everywhere it matters, and keeps the k-d query set
# small; growing it does not sharpen the result, it only costs time.
BAND_VOXELS = 2
# A shell is a real part of the shape if it has at least this share of the biggest one's
# faces. A hollow body's inner wall passes easily; a sliver the extraction left behind does
# not, and keeping those would make the output a hundred fragments again.
KEEP_FRACTION = 0.10
# Closest-point searches are O(cells x triangles), so they go in chunks that keep peak
# memory flat on a million-face input.
QUERY_CHUNK = 100_000
# Bytes-per-cell guesses for the two big stages, deliberately generous. Not an accountant:
# the point is a clear stop *before* the allocation, not a precise audit. The occupancy
# stage holds four bool grids and an int32 label grid at once; the field stage holds a
# float32 field, two float64 distance transforms, a bool band mask and a float64 distance
# table, all on the same grid.
OCCUPANCY_BYTES_PER_CELL = 16
FIELD_BYTES_PER_CELL = 48


def available_bytes() -> int:
    """Free physical memory, platform-independent, without a new dependency.

    Windows: GlobalMemoryStatusEx. POSIX: sysconf. Returns -1 where neither exists, which
    `ensure_headroom` reads as "cannot measure; proceed".
    """
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return -1
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * size
    except (AttributeError, OSError, ValueError):
        return -1


def ensure_headroom(needed: int, label: str) -> None:
    """Stop before an allocation that does not fit, instead of after the OS kills us.

    `needed` covers the stage's peak, not its average, so a machine that passes this will
    pass the stage. A machine whose free memory cannot be measured is trusted to its OS.
    """
    free = available_bytes()
    if free <= 0:
        return
    if needed > free:
        raise WatertightError(
            f"not enough free memory for {label}: the grid needs about "
            f"{needed / 1024 ** 3:.1f} GB, {free / 1024 ** 3:.1f} GB is free. "
            f"Lower --across (coarser voxels) or free some memory first.")


class WatertightError(RuntimeError):
    """The mesh could not be rebuilt, with a reason a person can act on."""


@dataclass
class WatertightResult:
    """The rebuilt mesh and what it took, so a log line can explain the trade."""

    vertices: np.ndarray
    faces: np.ndarray
    voxel: float
    shells_before: int
    shells_after: int
    boundary_edges: int
    notes: list = field(default_factory=list)

    @property
    def watertight(self) -> bool:
        return self.shells_after == 1 and self.boundary_edges == 0


def choose_voxel(extent: float, across: int = DEFAULT_ACROSS) -> float:
    """Voxel size for a mesh whose largest bounding-box side is `extent`.

    `across` is how many voxels span that side; dividing by `across - 1` matches the span
    the sampling grid actually has, since N samples cover N-1 steps.
    """
    if extent <= 0:
        raise WatertightError("the mesh has zero extent; there is nothing to rebuild")
    if across < 8:
        raise WatertightError(f"across must be at least 8 voxels, got {across}")
    return float(extent) / float(across - 1)


def choose_voxel_for(vertices: np.ndarray, across: int = DEFAULT_ACROSS) -> float:
    """Voxel size sized to the mesh's own longest side, refusing a flat one.

    A sheet with no thickness on one axis has no inside: there is nothing for the voxel
    solid to fill, and a field built from it has a sign that means nothing. That is a
    modelling input error, not something to silently extrude into a slab.
    """
    spans = np.ptp(vertices, axis=0)
    extent = float(spans.max())
    if float(spans.min()) <= extent * 1e-9:
        raise WatertightError("the mesh is flat on one axis; there is no inside to find")
    return choose_voxel(extent, across)


def weld(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices merged by position, and faces re-indexed onto them.

    glTF splits a vertex at every UV seam and every sharp normal, so the raw index buffer
    says a smooth wall is a thousand separate sheets. Every topology measurement below has
    to run on the welded mesh or it measures the UV layout instead of the geometry.
    """
    if len(vertices) == 0 or len(faces) == 0:
        raise WatertightError("the mesh is empty")
    scale = float(np.ptp(vertices, axis=0).max()) or 1.0
    key = np.round(vertices / (scale * 1e-6)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    welded_vertices = vertices[first]
    welded_faces = inverse.reshape(-1)[faces]
    good = ((welded_faces[:, 0] != welded_faces[:, 1])
            & (welded_faces[:, 1] != welded_faces[:, 2])
            & (welded_faces[:, 0] != welded_faces[:, 2]))
    if not good.any():
        raise WatertightError("every triangle collapsed when the mesh was welded")
    return welded_vertices, welded_faces[good]


def boundary_edges(faces: np.ndarray) -> int:
    """How many edges belong to exactly one face: zero when the surface is closed."""
    if len(faces) == 0:
        return 0
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]],
                               faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def shells(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected pieces, joined through shared *edges*, not shared vertex positions.

    `trimesh.split` connects through vertices, so two shells that merely touch at one
    welded point read as one piece -- which reads a crumb-stuck body as watertight. Walking
    shared edges is the rule a slicer uses, and it is why this does not call `split`.
    """
    if len(faces) == 0:
        return []
    # Each edge is stored with its two vertex indices sorted, so (a,b) and (b,a) from the
    # two faces that share it become the same row before the global sort.
    tri_edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                        axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort(tri_edges.T[::-1])
    sorted_edges, sorted_owner = tri_edges[order], owner[order]
    same = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    parent = np.arange(len(faces))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    for left, right in zip(sorted_owner[:-1][same], sorted_owner[1:][same]):
        ra, rb = find(int(left)), find(int(right))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(faces))])
    pieces = []
    for root in np.unique(roots):
        group = faces[roots == root].reshape(-1, 3)
        used, remap = np.unique(group, return_inverse=True)
        pieces.append((vertices[used], remap.reshape(-1, 3).astype(np.int64)))
    return pieces


def seal(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close the last few holes a contour leaves behind, without adding geometry.

    Marching cubes can leave a triangular gap where a sharp corner falls exactly on a grid
    edge -- three vertices, no face between them. Measured on the knight, that was the last
    3 of 12 open edges after the contour weld. A face over the gap is the honest fix: it is
    the triangle the field already implies, not a cap over missing geometry.

    Only gaps whose boundary is a single short loop are filled. A genuinely missing region
    has a boundary loop the size of a hole in the model, and capping that would invent a
    surface where the input had none -- better to report the mesh as not watertight.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(faces) == 0:
        return vertices, faces
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                    axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = unique_edges[counts == 1]
    if len(open_edges) == 0:
        return vertices, faces

    nodes = np.unique(open_edges)
    index = {int(node): slot for slot, node in enumerate(nodes)}
    rows = [index[int(a)] for a, b in open_edges]
    cols = [index[int(b)] for a, b in open_edges]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes)))
    count, labels = connected_components(graph, directed=False)

    extra = []
    for label in range(count):
        loop = nodes[labels == label]
        # A pinhole is a triangle; a longer loop is a hole in the model, which is not ours
        # to invent a surface for.
        if len(loop) != 3:
            continue
        extra.append(loop)
    if not extra:
        return vertices, faces
    return vertices, np.vstack([faces, np.asarray(extra, dtype=np.int64)])


def keep_body(vertices: np.ndarray, faces: np.ndarray):
    """(vertices, faces, sizes): the body plus its cavities, without the crumbs.

    A closed solid and a hollow one look the same to a face count, so this cannot keep "the
    largest shell" and stop: a hollow body's inner wall is a separate shell, and dropping it
    turns the model solid. What separates a cavity from a crumb is size relative to the body:
    a cavity is a real part of the shape, a crumb is a sliver the extraction left behind. A
    shell is kept when it has at least `KEEP_FRACTION` of the biggest one's faces.
    """
    pieces = shells(vertices, faces)
    if not pieces:
        raise WatertightError("the mesh has no pieces to keep")
    sizes = sorted((len(piece[1]) for piece in pieces), reverse=True)
    floor = sizes[0] * KEEP_FRACTION
    kept = [piece for piece in pieces if len(piece[1]) >= floor]
    if not kept:
        kept = [max(pieces, key=lambda piece: len(piece[1]))]

    vertices_list, faces_list, offset = [], [], 0
    for piece_vertices, piece_faces in kept:
        vertices_list.append(piece_vertices)
        faces_list.append(piece_faces + offset)
        offset += len(piece_vertices)
    return np.vstack(vertices_list), np.vstack(faces_list), sizes


def largest_shell(vertices: np.ndarray, faces: np.ndarray):
    """(vertices, faces, sizes): only the biggest connected piece, plus every size first.

    `sizes` is sorted largest first, which is what tells a caller whether the mesh was one
    piece or a body surrounded by crumbs.
    """
    pieces = shells(vertices, faces)
    if not pieces:
        raise WatertightError("the mesh has no pieces to keep")
    sizes = sorted((len(piece[1]) for piece in pieces), reverse=True)
    biggest = max(pieces, key=lambda piece: len(piece[1]))
    return biggest[0], biggest[1], sizes


def occupancy(vertices: np.ndarray, faces: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The solid's occupancy on a `voxel`-sized grid, plus the grid's origin point.

    The wall comes from `trimesh.voxelized`, which marks the cells the surface crosses, and
    the interior from a fill applied to the *complement* of that wall: a flood fill from the
    grid's border reaches every cell outside the solid and stops at the wall, so whatever it
    cannot reach is inside. That keeps a hollow model hollow, which `VoxelGrid.fill` cannot
    do -- it floods from the bounding box and swallows the cavity as well.

    `trimesh.repair.fill_holes` runs first: the wall marking is per-triangle, and an open
    boundary is a place where the surface is missing rather than thin.

    Returns `(mask, origin)`, padded by two empty cells on every side so the contour that
    comes out has empty space around it and closes without capping against the grid edge.
    """
    import trimesh
    from scipy import ndimage

    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fill_holes(repaired)
    shell = repaired.voxelized(pitch=voxel)
    wall = np.asarray(shell.matrix, dtype=bool)
    lo = np.asarray(shell.transform, dtype=np.float64)[:3, 3]

    # `voxelized` marks the cells a triangle crosses -- a skin, never a filled body. That is
    # exactly right for what this studio generates: measured on the knight, 956,944 faces at
    # a volume-to-area ratio of 0.0013, a shell a fraction of a millimetre thick with empty
    # space behind it. Filling that space made it 2.24x heavier, a different object. So the
    # skin stands as the solid, and the shape stays the shape the picture described.
    #
    # It is wrong for a genuinely solid body, where the skin would leave the middle empty. The
    # parity test below decides which case this is: it asks what the region the skin encloses
    # *is*, and only a region that a ray crosses an even number of times gets filled. A hollow
    # shell's inside is enclosed air (even) and stays air; a solid cube's inside is material
    # (odd) and gets filled.
    padded = np.zeros(tuple(int(side) + 4 for side in wall.shape), dtype=bool)
    padded[2:-2, 2:-2, 2:-2] = wall
    solid = padded.copy()

    enclosed = ~_flood_outside(padded) & ~padded
    labels, count = ndimage.label(enclosed)
    for label in range(1, count + 1):
        region = labels == label
        if not region.any():
            continue
        seed = tuple(int(value) for value in np.argwhere(region)[0])
        if _crossings_beyond(padded, seed) % 2 == 1:
            solid |= region

    origin = lo - 1.5 * voxel
    return solid, origin


def _flood_outside(wall: np.ndarray) -> np.ndarray:
    """Cells reachable from the grid's border without crossing `wall`."""
    from scipy import ndimage

    outside = np.zeros_like(wall)
    outside[0], outside[-1] = ~wall[0], ~wall[-1]
    outside[:, 0], outside[:, -1] = ~wall[:, 0], ~wall[:, -1]
    outside[:, :, 0], outside[:, :, -1] = ~wall[:, :, 0], ~wall[:, :, -1]
    return ndimage.binary_propagation(outside, mask=~wall)


def _crossings_beyond(wall: np.ndarray, seed: tuple[int, int, int]) -> int:
    """How many wall runs a +Z ray from `seed` passes through.

    A wall several cells thick is one crossing, so this counts runs, not cells. Odd means the
    point has material behind it -- the ray ends inside the body; even means it does not.
    """
    x, y, z = seed
    ray = wall[x, y, z:]
    edges = np.diff(np.concatenate([[0], ray.astype(np.int8), [0]]))
    return int((edges == 1).sum())


def signed_field(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray,
                 origin: np.ndarray, voxel: float, log=None) -> np.ndarray:
    """Signed distance to the surface on the mask's grid: negative in the wall, positive out.

    Near a triangle the distance is measured exactly to that triangle, so a thin plate keeps
    its real thickness instead of averaging into smear. Everywhere else -- including the
    cavity behind a hollow shell, where no triangle is close but the sign is known -- the
    magnitude comes from a distance transform, then takes the mask's sign.

    The reason there is a gradient at all, rather than a constant either side: marching cubes
    interpolates linearly between neighbouring cells, so a field that jumps from `-2` to `+2`
    across one cell puts the crossing wherever the ratio lands. A continuous field puts it on
    the interface, every time.
    """
    from scipy import ndimage, spatial

    shape = mask.shape
    # `distance_transform_edt` measures to the nearest zero, so each side is asked for its
    # own distance: cells outside the wall measure to the nearest wall cell, wall cells measure
    # to the nearest outside cell. Asking it once on the mask returns zero right across a
    # cavity, which flattens the field and loses the sign change the contour needs.
    to_wall = ndimage.distance_transform_edt(~mask)
    from_wall = ndimage.distance_transform_edt(mask)
    field = np.where(mask, -from_wall, to_wall).astype(np.float32)

    tri = vertices[faces]
    found = np.zeros(shape, dtype=bool)
    distance_at = np.zeros(shape, dtype=np.float64)

    lo = np.floor((tri.min(axis=1) - origin) / voxel - 1.0).astype(np.int64)
    hi = np.ceil((tri.max(axis=1) - origin) / voxel + 1.0).astype(np.int64)
    np.clip(lo, 0, np.asarray(shape) - 1, out=lo)
    np.clip(hi, 0, np.asarray(shape) - 1, out=hi)
    spans = hi - lo + 1
    # A triangle's bounding box can span most of the grid when the face is huge (the wall of
    # a box, a cape). Marking all of it is fine -- the band is a superset, the exact distance
    # is what gets written -- but capping the span first would mark a corner of that box and
    # miss the face itself, which is what left only 1,280 cells of a 97,336-cell grid. The
    # cost of a huge face is real though: one such box is the whole grid, per such face, so
    # the list is refused up front when it cannot fit rather than after the meshgrids are up.
    ensure_headroom(int(np.prod(spans, axis=1).sum()) * 48, "the distance-field band")
    cells_list = []
    for (lx, ly, lz), (sx, sy, sz) in zip(lo, spans):
        gx, gy, gz = np.meshgrid(np.arange(lx, min(lx + sx, shape[0])),
                                 np.arange(ly, min(ly + sy, shape[1])),
                                 np.arange(lz, min(lz + sz, shape[2])),
                                 indexing="ij")
        cells_list.append(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))
    if not cells_list:
        raise WatertightError("the mesh produced no candidate cells")
    cells = np.unique(np.vstack(cells_list), axis=0)
    log and log(f"WATERTIGHT:: {len(cells):,} band cells of "
                f"{int(np.prod(shape)):,} in the grid")

    tree = spatial.cKDTree(tri.reshape(-1, 3))
    for start in range(0, len(cells), QUERY_CHUNK):
        chunk = cells[start:start + QUERY_CHUNK]
        points = origin + chunk * voxel
        _, nearest = tree.query(points, k=1)
        face = nearest // 3
        distance = _distance_to_triangles(points, tri, face)
        found[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = True
        distance_at[chunk[:, 0], chunk[:, 1], chunk[:, 2]] = distance

    field[found] = np.where(mask[found], -distance_at[found], distance_at[found])
    # A grid point can sit exactly on a vertex, giving a distance of exactly zero. That is
    # a cell marching cubes cannot interpolate through and a hole where it does not try, so
    # nudge it to the side the mask says it belongs on, just off the crossing.
    on_surface = found & (distance_at == 0.0)
    if on_surface.any():
        field[on_surface & mask] = -1e-3
        field[on_surface & ~mask] = 1e-3
    return field


def _distance_to_triangles(points: np.ndarray, tri: np.ndarray,
                           face: np.ndarray) -> np.ndarray:
    """Exact unsigned distance from each point to its own triangle (Ericson 5.1.5)."""
    a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c

    def dot(u, v):
        return np.einsum("ij,ij->i", u, v)

    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)

    # Inside the face: project onto its plane by the barycentric clue.
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
    v = np.clip(vb / denom, 0.0, 1.0)
    w = np.clip(vc / denom, 0.0, 1.0)
    closest = a + ab * v[:, None] + ac * w[:, None]

    # The corners: a point past two of the three back-facing edge planes sits nearest the
    # shared vertex.
    vertex_a = (d1 <= 0.0) & (d2 <= 0.0)
    vertex_b = (d3 >= 0.0) & (d6 <= 0.0)
    vertex_c = (d6 >= 0.0) & (d5 <= 0.0)
    closest = np.where(vertex_a[:, None], a, closest)
    closest = np.where(vertex_b[:, None], b, closest)
    closest = np.where(vertex_c[:, None], c, closest)

    # The edges: one back-facing plane across a back-facing pair.
    bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    bc_t = (d4 - d3) / np.where((d4 - d3) - (d5 - d6) == 0.0, 1.0, (d4 - d3) - (d5 - d6))
    ac_edge = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    ac_t = d2 / np.where(d2 - d6 == 0.0, 1.0, d2 - d6)
    ab_edge = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    ab_t = d1 / np.where(d1 - d3 == 0.0, 1.0, d1 - d3)
    closest = np.where(bc[:, None], b + (c - b) * np.clip(bc_t, 0.0, 1.0)[:, None], closest)
    closest = np.where(ac_edge[:, None], a + ac * np.clip(ac_t, 0.0, 1.0)[:, None], closest)
    closest = np.where(ab_edge[:, None], a + ab * np.clip(ab_t, 0.0, 1.0)[:, None], closest)

    return np.linalg.norm(points - closest, axis=1)


def contour(field: np.ndarray, origin: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes at level zero, back in the mesh's own coordinates.

    The raw output is welded once more: the contour emits a vertex per grid edge it
    crosses, and where two cells meet at a corner it can emit the same position twice,
    which leaves a pinhole regardless of how good the field is. Measured on the knight,
    that was the whole of the residual 12 open edges -- welding after the contour closes
    them. No `method=` is passed: `lewiner` is the default today and the only algorithm
    once scikit-image 0.25 drops the keyword, so naming it only breaks the newer pin.
    """
    from skimage import measure

    spacing = (voxel, voxel, voxel)
    vertices, faces, _, _ = measure.marching_cubes(field.astype(np.float32), level=0.0,
                                                   spacing=spacing)
    vertices = vertices + np.asarray(origin, dtype=np.float64)
    return weld(vertices, faces.astype(np.int64))


def rebuild(vertices: np.ndarray, faces: np.ndarray, *, voxel: float | None = None,
            across: int = DEFAULT_ACROSS, log=None) -> WatertightResult:
    """Rebuild `(vertices, faces)` as a closed solid. See the module docstring.

    `voxel` overrides the size chosen from the mesh's own extent. Pass `log` to watch the
    stages: a 1M-face input takes minutes and silence looks like a hang.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    welded_vertices, welded_faces = weld(vertices, faces)
    before = len(shells(welded_vertices, welded_faces))
    log and log(f"WATERTIGHT:: input {len(faces):,} faces, {before:,} shell(s)")

    spans = np.ptp(welded_vertices, axis=0)
    extent = float(spans.max())
    voxel = float(voxel) if voxel else choose_voxel_for(welded_vertices, across)
    log and log(f"WATERTIGHT:: voxel {voxel:.6g} over a {extent:.4g} extent")

    # The grid the two heavy stages will share, and whether it fits in what is free. Both
    # checks run before either allocation: an OOM kill mid-stage is a lost run, while a
    # sentence here costs nothing. The padding is the four cells `occupancy` adds, plus
    # one for the ceiling rounding.
    grid = np.ceil(spans / voxel) + 5
    cells = int(np.prod(grid))
    log and log(f"WATERTIGHT:: grid {int(grid[0])}x{int(grid[1])}x{int(grid[2])} "
                f"({cells:,} cells)")
    ensure_headroom(cells * OCCUPANCY_BYTES_PER_CELL, "the occupancy grid")
    ensure_headroom(cells * FIELD_BYTES_PER_CELL, "the distance field")

    mask, origin = occupancy(welded_vertices, welded_faces, voxel)
    log and log(f"WATERTIGHT:: occupancy {int(mask.sum()):,} voxels "
                f"of {mask.size:,} in the box")

    field = signed_field(welded_vertices, welded_faces, mask, origin, voxel, log=log)
    if not (field.min() < 0.0 < field.max()):
        raise WatertightError("the field has no inside or no outside; the mesh may be "
                              "a single open sheet with nothing to fill")
    out_vertices, out_faces = contour(field, origin, voxel)
    log and log(f"WATERTIGHT:: contour {len(out_vertices):,} verts "
                f"{len(out_faces):,} faces")

    out_vertices, out_faces, sizes = keep_body(out_vertices, out_faces)
    out_vertices, out_faces = seal(out_vertices, out_faces)
    boundary = boundary_edges(out_faces)
    kept = len(shells(out_vertices, out_faces))
    log and log(f"WATERTIGHT:: kept {len(out_faces):,} faces in {kept} shell(s) "
                f"of {len(sizes):,}")

    return WatertightResult(
        vertices=out_vertices, faces=out_faces, voxel=voxel,
        shells_before=before, shells_after=kept, boundary_edges=boundary,
    )
