import math
import bmesh
from mathutils import Vector


def _round_vec(co, inv_eps):
    return (
        int(round(co.x * inv_eps)),
        int(round(co.y * inv_eps)),
        int(round(co.z * inv_eps)),
    )


def _face_bbox(coords):
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    zs = [c.z for c in coords]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _bbox_overlap(a, b, margin=0.0):
    return (
        a[0] - margin <= b[3] + margin
        and a[3] + margin >= b[0] - margin
        and a[1] - margin <= b[4] + margin
        and a[4] + margin >= b[1] - margin
        and a[2] - margin <= b[5] + margin
        and a[5] + margin >= b[2] - margin
    )


def find_non_manifold_edges(bm):
    """Рёбра с числом смежных граней != 2 (включая wire и T-junctions)."""
    bm.edges.ensure_lookup_table()
    return [e for e in bm.edges if e.is_valid and not e.is_manifold]


def find_wire_edges(bm):
    """Рёбра без смежных граней."""
    bm.edges.ensure_lookup_table()
    return [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]


def find_non_manifold_vertices(bm):
    """Вершины, через которые проходит «ломаная» топология."""
    bm.verts.ensure_lookup_table()
    return [v for v in bm.verts if v.is_valid and not v.is_manifold]


def find_boundary_edges(bm):
    bm.edges.ensure_lookup_table()
    return [e for e in bm.edges if e.is_valid and e.is_boundary]


def find_boundary_loops(bm):
    """
    Группирует граничные рёбра в замкнутые контуры (дыры).
    Возвращает список контуров; каждый контур — список рёбер.
    """
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    remaining = {e for e in bm.edges if e.is_valid and e.is_boundary}
    loops = []

    while remaining:
        start = remaining.pop()
        loop = [start]
        tip = start.verts[1]
        prev = start.verts[0]
        origin = prev

        while True:
            next_edge = None
            for e in tip.link_edges:
                if not e.is_boundary or e not in remaining:
                    continue
                other = e.other_vert(tip)
                if other == prev:
                    continue
                next_edge = e
                break

            if next_edge is None:
                break

            remaining.remove(next_edge)
            loop.append(next_edge)
            prev, tip = tip, next_edge.other_vert(tip)
            if tip == origin:
                break

        loops.append(loop)

    return loops


def find_loose_vertices(bm):
    return [v for v in bm.verts if v.is_valid and not v.link_edges]


def find_degenerate_faces(bm, min_area=1e-10, min_edge_length=1e-8):
    bm.faces.ensure_lookup_table()
    degenerate = []

    for f in bm.faces:
        if not f.is_valid:
            continue
        if len(f.verts) < 3:
            degenerate.append(f)
            continue

        verts = f.verts
        for i in range(len(verts)):
            a = verts[i].co
            b = verts[(i + 1) % len(verts)].co
            if (a - b).length_squared <= min_edge_length * min_edge_length:
                degenerate.append(f)
                break
        else:
            try:
                if f.calc_area() <= min_area:
                    degenerate.append(f)
            except (RuntimeError, ValueError):
                degenerate.append(f)

    return degenerate


def find_duplicate_vertices(bm, distance=0.0001):
    """
    Поиск дубликатов с учётом соседних ячеек сетки (нет ложных пропусков на границах).
    """
    bm.verts.ensure_lookup_table()

    if not bm.verts or distance <= 0:
        return [], 0

    inv = 1.0 / distance
    dist_sq = distance * distance
    grid = {}

    for v in bm.verts:
        if not v.is_valid:
            continue
        ix = int(math.floor(v.co.x * inv))
        iy = int(math.floor(v.co.y * inv))
        iz = int(math.floor(v.co.z * inv))
        grid.setdefault((ix, iy, iz), []).append(v)

    parent = {}

    def find(v):
        parent.setdefault(v, v)
        if parent[v] is not v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra is not rb:
            parent[rb] = ra

    for (ix, iy, iz), cell_verts in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = grid.get((ix + dx, iy + dy, iz + dz))
                    if not neighbor:
                        continue
                    for va in cell_verts:
                        for vb in neighbor:
                            if va is vb:
                                continue
                            if (va.co - vb.co).length_squared <= dist_sq:
                                union(va, vb)

    groups = {}
    for v in parent:
        root = find(v)
        groups.setdefault(root, []).append(v)

    duplicate_verts = []
    duplicate_groups = 0
    for group in groups.values():
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_verts.extend(group)

    return duplicate_verts, duplicate_groups


def find_flipped_normals(bm, angle_threshold=0.0):
    """
    Сравнивает нормали с пересчитанными «наружу» на копии BMesh (без зависимости от context).
    angle_threshold в радианах: грани с углом больше порога считаются перевёрнутыми.
    """
    if not bm.faces:
        return []

    bm.faces.ensure_lookup_table()
    bm.faces.index_update()

    temp_bm = bm.copy()
    try:
        temp_bm.faces.ensure_lookup_table()
        temp_bm.faces.index_update()
        temp_bm.normal_update()
        bmesh.ops.recalc_face_normals(temp_bm, faces=temp_bm.faces)

        flipped = []
        dot_threshold = math.cos(math.pi - angle_threshold) if angle_threshold > 0 else 0.0

        for f_orig in bm.faces:
            if not f_orig.is_valid or len(f_orig.verts) < 3:
                continue
            try:
                f_temp = temp_bm.faces[f_orig.index]
                if not f_temp.is_valid:
                    continue
                dot = f_orig.normal.dot(f_temp.normal)
                if dot < dot_threshold:
                    flipped.append(f_orig)
            except (IndexError, RuntimeError):
                continue

        return flipped
    finally:
        temp_bm.free()


def find_overlapping_faces(bm, eps=1e-5):
    """
    Дубликаты полигонов по геометрической сигнатуре (порядок вершин не важен).
    """
    bm.faces.ensure_lookup_table()
    inv_eps = 1.0 / eps if eps > 0 else 1e5
    signature_map = {}
    overlapping = []

    for f in bm.faces:
        if not f.is_valid or len(f.verts) < 3:
            continue

        key = tuple(sorted(_round_vec(v.co, inv_eps) for v in f.verts))
        if key in signature_map:
            overlapping.append(f)
            overlapping.append(signature_map[key])
        else:
            signature_map[key] = f

    return overlapping


def _is_point_in_triangle(p, a, b, c, eps=1e-6):
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = v0.dot(v0)
    dot01 = v0.dot(v1)
    dot02 = v0.dot(v2)
    dot11 = v1.dot(v1)
    dot12 = v1.dot(v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= -eps) and (v >= -eps) and (u + v <= 1 + eps)


def _seg_tri_intersect(p1, p2, t1, t2, t3, eps=1e-6):
    edge1 = t2 - t1
    edge2 = t3 - t1
    normal = edge1.cross(edge2)

    if normal.length < eps:
        return False

    normal.normalize()

    dist1 = (p1 - t1).dot(normal)
    dist2 = (p2 - t1).dot(normal)

    if dist1 * dist2 > eps * eps:
        return False

    t_val = dist1 / (dist1 - dist2 + 1e-12)
    if t_val < -eps or t_val > 1.0 + eps:
        return False

    intersection_point = p1 + (p2 - p1) * t_val
    return _is_point_in_triangle(intersection_point, t1, t2, t3, eps)


def check_triangles_intersection(tri1, tri2, eps=1e-6):
    edges1 = [(tri1[0], tri1[1]), (tri1[1], tri1[2]), (tri1[2], tri1[0])]
    for p1, p2 in edges1:
        if _seg_tri_intersect(p1, p2, tri2[0], tri2[1], tri2[2], eps):
            return True

    edges2 = [(tri2[0], tri2[1]), (tri2[1], tri2[2]), (tri2[2], tri2[0])]
    for p1, p2 in edges2:
        if _seg_tri_intersect(p1, p2, tri1[0], tri1[1], tri1[2], eps):
            return True

    return False


def _triangulate_fan(coords):
    if len(coords) < 3:
        return []
    if len(coords) == 3:
        return [tuple(coords)]
    return [(coords[0], coords[k], coords[k + 1]) for k in range(1, len(coords) - 1)]


def find_intersecting_faces(bm, eps=1e-6):
    bm.faces.ensure_lookup_table()
    valid_faces = [f for f in bm.faces if f.is_valid and len(f.verts) >= 3]
    if len(valid_faces) < 2:
        return []

    face_data = []
    for f in valid_faces:
        coords = [v.co for v in f.verts]
        tris = _triangulate_fan(coords)
        face_data.append((f, tris, _face_bbox(coords), set(f.verts)))

    intersecting = set()
    n = len(face_data)

    for i in range(n):
        f1, tris1, bb1, verts1 = face_data[i]
        for j in range(i + 1, n):
            f2, tris2, bb2, verts2 = face_data[j]

            if verts1 & verts2:
                continue
            if not _bbox_overlap(bb1, bb2, margin=eps):
                continue

            for t1 in tris1:
                hit = False
                for t2 in tris2:
                    if check_triangles_intersection(t1, t2, eps):
                        intersecting.add(f1)
                        intersecting.add(f2)
                        hit = True
                        break
                if hit:
                    break

    return list(intersecting)


def collect_hole_fill_edges(bm, max_sides=0, selected_indices=None):
    """
    Собирает граничные рёбра для одного вызова holes_fill.
    max_sides передаётся в holes_fill (0 = без лимита).
    selected_indices: если задано, только контуры с выделенным ребром.
    """
    loops = find_boundary_loops(bm)
    if not loops:
        return [], 0

    target_loops = []
    for loop in loops:
        if len(loop) < 3:
            continue
        if selected_indices is not None:
            if not any(e.index in selected_indices for e in loop):
                continue
        target_loops.append(loop)

    seen = set()
    edges = []
    for loop in target_loops:
        for e in loop:
            if e.is_valid and e.index not in seen:
                seen.add(e.index)
                edges.append(e)

    return edges, len(target_loops)
