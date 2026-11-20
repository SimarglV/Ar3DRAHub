"""The Watertight step: topology measurement, and the rebuild itself on shapes whose
right answer is known. No Blender, no GPU, no model weights."""

from __future__ import annotations


import numpy as np
import pytest

from Ar3DRAHub import watertight as wt


def cube(size: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A closed cube as 12 triangles, one shell, zero boundary edges."""
    import trimesh

    mesh = trimesh.creation.box(extents=(size, size, size))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def open_sheet() -> tuple[np.ndarray, np.ndarray]:
    """A flat quad: every edge is a boundary, and there is no inside to find."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def two_cubes() -> tuple[np.ndarray, np.ndarray]:
    """A big cube and a small detached one: two shells, both closed."""
    v1, f1 = cube(2.0)
    v2, f2 = cube(0.4)
    v2 = v2 + np.array([4.0, 0.0, 0.0])
    return (np.vstack([v1, v2]),
            np.vstack([f1, f2 + len(v1)]))


# --- the measurements -------------------------------------------------------------------

def test_a_cube_is_closed():
    vertices, faces = cube()
    assert wt.boundary_edges(faces) == 0


def test_an_open_sheet_is_not():
    _, faces = open_sheet()
    # Two triangles sharing a diagonal: the four rim edges are used once.
    assert wt.boundary_edges(faces) == 4


def test_a_cube_is_one_shell():
    vertices, faces = cube()
    assert len(wt.shells(vertices, faces)) == 1


def test_two_detached_cubes_are_two_shells():
    vertices, faces = two_cubes()
    assert len(wt.shells(vertices, faces)) == 2


def test_largest_shell_keeps_the_body_and_reports_both_sizes():
    vertices, faces = two_cubes()
    _, kept_faces, sizes = wt.largest_shell(vertices, faces)
    assert sizes == [12, 12]        # both cubes, largest first
    assert len(kept_faces) == 12


def test_shells_keep_a_point_welded_pair_apart():
    """Two tetrahedra sharing exactly one vertex, and no edge or face.

    `trimesh.split` follows the face-adjacency graph, so whether it joins a point-welded
    pair depends on how it counts a vertex used once by each side. This measurement never
    looks at vertices at all -- it walks shared edges, which is the rule a slicer uses.
    """

    top = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
    top_faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    bottom = top * np.array([1.0, 1.0, -1.0])
    vertices = np.vstack([top, bottom])
    faces = np.vstack([top_faces, top_faces + len(top)])

    welded_vertices, welded_faces = wt.weld(vertices, faces)
    assert len(welded_vertices) == 7                 # the shared tip merged
    assert len(wt.shells(welded_vertices, welded_faces)) == 2


def test_weld_merges_seams_and_drops_collapsed_triangles():
    vertices, faces = cube()
    # Split every vertex per-corner, exactly as glTF does at a UV seam.
    split = vertices[faces.reshape(-1)]
    split_faces = np.arange(len(split), dtype=np.int64).reshape(-1, 3)
    welded_vertices, welded_faces = wt.weld(split, split_faces)
    assert len(welded_vertices) == 8
    assert len(welded_faces) == 12
    assert wt.boundary_edges(welded_faces) == 0


def test_the_seam_split_cube_measures_as_broken_before_welding():
    """The reason welding comes first: on the split mesh every edge is a boundary."""
    vertices, faces = cube()
    split_faces = np.arange(36, dtype=np.int64).reshape(-1, 3)
    assert wt.boundary_edges(split_faces) == 36


# --- voxel size -------------------------------------------------------------------------

def test_choose_voxel_spans_the_extent_in_the_requested_steps():
    assert wt.choose_voxel(1.0, across=101) == pytest.approx(0.01)


def test_choose_voxel_refuses_a_flat_mesh():
    with pytest.raises(wt.WatertightError, match="zero extent"):
        wt.choose_voxel(0.0)


def test_choose_voxel_refuses_a_grid_too_small_to_hold_a_shape():
    with pytest.raises(wt.WatertightError, match="at least 8"):
        wt.choose_voxel(1.0, across=4)


# --- the rebuild ------------------------------------------------------------------------

def test_rebuild_closes_a_cube_into_one_shell():
    vertices, faces = cube()
    result = wt.rebuild(vertices, faces, across=32)
    assert result.watertight, result.notes
    assert result.boundary_edges == 0
    assert result.shells_after == 1


def test_rebuild_keeps_only_the_body_from_a_cube_and_a_crumb():
    vertices, faces = two_cubes()
    result = wt.rebuild(vertices, faces, across=64)
    assert result.shells_before == 2
    assert result.shells_after == 1
    # The crumb's volume is 1/125 of the body's, so the body is what survives.
    lo, hi = result.vertices.min(axis=0), result.vertices.max(axis=0)
    assert float(hi[0] - lo[0]) > 1.0


def test_rebuild_refuses_a_single_open_sheet():
    vertices, faces = open_sheet()
    with pytest.raises(wt.WatertightError, match="no inside"):
        wt.rebuild(vertices, faces, across=32)


def test_rebuild_refuses_an_empty_mesh():
    with pytest.raises(wt.WatertightError, match="empty"):
        wt.rebuild(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))


def test_rebuild_reports_the_voxel_it_used():
    vertices, faces = cube()
    result = wt.rebuild(vertices, faces, voxel=0.1)
    assert result.voxel == pytest.approx(0.1)
    assert result.watertight


def test_available_bytes_measures_something_or_says_it_cannot():
    """-1 is the honest 'cannot measure' answer; anything else is a real number."""
    value = wt.available_bytes()
    assert value == -1 or value > 0


def test_the_rebuild_stops_before_the_grid_when_memory_is_short(monkeypatch):
    """The requirement is a clear sentence, not an OOM kill mid-stage. With a kilobyte
    'free', the first headroom check must refuse -- the grid is bigger than that."""
    monkeypatch.setattr(wt, "available_bytes", lambda: 1024)
    vertices, faces = cube()
    with pytest.raises(wt.WatertightError, match="not enough free memory"):
        wt.rebuild(vertices, faces, across=32)


def test_the_rebuild_does_not_warn_on_a_machine_it_cannot_measure(monkeypatch):
    """'Cannot measure' means trust the OS, not refuse everything."""
    monkeypatch.setattr(wt, "available_bytes", lambda: -1)
    vertices, faces = cube()
    result = wt.rebuild(vertices, faces, across=32)
    assert result.watertight


def test_the_band_guard_refuses_a_grid_sized_band(monkeypatch):
    """A huge triangle's bounding box is the whole grid: `signed_field` must refuse to
    build that list when it cannot fit, the same way `rebuild` refuses its grids."""
    monkeypatch.setattr(wt, "available_bytes", lambda: 1024)
    # One giant triangle spanning the whole 3x3x3 extent: its voxel-space box is the grid.
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(wt.WatertightError, match="not enough free memory"):
        wt.signed_field(vertices, faces,
                        mask=np.zeros((12, 12, 12), dtype=bool),
                        origin=np.zeros(3), voxel=0.3)


# --- the last few holes -----------------------------------------------------------------

def test_seal_closes_a_single_triangle_pinhole():
    """A cube with one face removed leaves a 4-vertex loop, not a pinhole; a cube with a
    triangle's worth of edges open is what marching cubes actually leaves. Build the
    smallest case: a closed cube minus one face, which seal must NOT cap."""
    vertices, faces = cube()
    holed = faces[1:]                       # one triangle gone
    sealed_v, sealed_f = wt.seal(vertices, holed)
    assert len(sealed_f) == len(faces)      # its 3-vertex loop is a pinhole, so it closes
    assert wt.boundary_edges(sealed_f) == 0


def test_seal_leaves_a_real_hole_alone():
    """A whole box face is missing: 4 boundary vertices in one loop. That is a hole in the
    model, and capping it would invent a surface the input never had."""
    vertices, faces = cube()
    without_top = faces[2:]                 # two triangles gone: one quad-sized opening
    sealed_v, sealed_f = wt.seal(vertices, without_top)
    assert len(sealed_f) == len(without_top)
    assert wt.boundary_edges(sealed_f) > 0


def test_seal_is_a_no_op_on_a_closed_mesh():
    vertices, faces = cube()
    _, sealed_f = wt.seal(vertices, faces)
    assert len(sealed_f) == len(faces)


# --- hollow models stay hollow ----------------------------------------------------------

def hollow_box(wall: float = 0.2, size: float = 2.0):
    """A closed box with a closed cavity inside it: two shells, volume = the wall only."""
    import trimesh

    outer = trimesh.creation.box(extents=(size, size, size))
    inner = trimesh.creation.box(extents=(size - 2 * wall,) * 3)
    inner.invert()
    joined = trimesh.util.concatenate([outer, inner])
    return np.asarray(joined.vertices, dtype=np.float64), np.asarray(joined.faces,
                                                                     dtype=np.int64)


def test_occupancy_keeps_a_cavity_hollow():
    """The bug this exists for: `VoxelGrid.fill` floods from the bounding box and swallows
    the cavity, so a hollow model came out solid and 2.4x its real volume."""
    vertices, faces = hollow_box()
    welded_v, welded_f = wt.weld(vertices, faces)
    mask, _ = wt.occupancy(welded_v, welded_f, 0.05)
    centre = tuple(int(side) // 2 for side in mask.shape)
    assert not mask[centre], "the cavity was filled in"


def test_occupancy_still_fills_a_solid_body():
    """The other half: an ordinary closed cube has no cavity, and must read solid."""
    vertices, faces = cube()
    welded_v, welded_f = wt.weld(vertices, faces)
    mask, _ = wt.occupancy(welded_v, welded_f, 0.05)
    centre = tuple(int(side) // 2 for side in mask.shape)
    assert mask[centre]


def test_rebuild_of_a_hollow_box_keeps_the_wall_not_the_whole_block():
    """Volume is the check: 8.0 solid, 3.904 wall.

    Marching cubes places the surface on the cell boundary, so the cavity comes out one
    voxel smaller per axis than it is -- measured 5.15 at `across=40` (voxel 0.0513). The
    band is wide on purpose: it must not be 8.0, which is what a filled cavity gives.
    """
    vertices, faces = hollow_box()
    result = wt.rebuild(vertices, faces, across=40)
    mesh = __import__("trimesh").Trimesh(vertices=result.vertices, faces=result.faces,
                                         process=False)
    assert 3.4 < mesh.volume < 5.6, mesh.volume
    assert mesh.is_watertight


def test_keep_body_keeps_a_cavity_but_drops_a_crumb():
    """The cavity is ~40% of the body here; a sliver is far below the cut-off."""
    vertices, faces = hollow_box()
    result = wt.rebuild(vertices, faces, across=40)
    pieces = wt.shells(result.vertices, result.faces)
    assert len(pieces) == 2, [len(piece[1]) for piece in pieces]
