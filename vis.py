import pygame
import sys
import random
from itertools import combinations
from shapely.geometry import Polygon, box

# --- Config ---
r = 1
NUM_POINTS = 3 * r
WIDTH, HEIGHT = 1600, 1200
RADIUS = 10

# --- Colors ---
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)
TRANSPARENT_BLUE_BASE_ALPHA = 40

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("All Triangle Partitions and Intersections")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)

# --- Triangle Partition Generator ---
def triangle_partitions(indices):
    if len(indices) % 3 != 0:
        return []
    def helper(remaining):
        if not remaining:
            yield []
            return
        first = remaining[0]
        for pair in combinations(remaining[1:], 2):
            group = [first, *pair]
            rest = [i for i in remaining if i not in group]
            for sub in helper(rest):
                yield [group] + sub
    return list(helper(indices))

# --- Points ---
def generate_random_points():
    return [{
        "pos": pygame.Vector2(random.randint(100, WIDTH - 100),
                              random.randint(100, HEIGHT - 100)),
        "label": f"P{i}"
    } for i in range(NUM_POINTS)]

# --- Cache management ---
def update_partitions(r_value):
    global NUM_POINTS, all_indices, all_partitions
    NUM_POINTS = 3 * r_value
    all_indices = list(range(NUM_POINTS))
    all_partitions = triangle_partitions(all_indices)
    print(f"Total partitions: {len(all_partitions)}")

update_partitions(r)
points = generate_random_points()

# Map each triangle (tuple of indices) to its polygon and bounding box
triangle_cache = {}  # key: tuple(sorted points), value: (Polygon, bounds)
# Map partitions to list of triangle keys
partition_triangles_keys = []  # list of lists of triangle keys

def build_triangle_cache_and_partitions():
    global triangle_cache, partition_triangles_keys
    triangle_cache.clear()
    partition_triangles_keys.clear()
    for partition in all_partitions:
        tri_keys = []
        for tri in partition:
            key = tuple(sorted(tri))
            if key not in triangle_cache:
                pts = [points[i]["pos"] for i in key]
                poly = Polygon([(p.x, p.y) for p in pts])
                triangle_cache[key] = (poly, poly.bounds)  # bounds = (minx, miny, maxx, maxy)
            tri_keys.append(key)
        partition_triangles_keys.append(tri_keys)

build_triangle_cache_and_partitions()

def update_triangles_for_point(pt_index):
    # Update cached polygons that include pt_index
    keys_to_update = [key for key in triangle_cache if pt_index in key]
    for key in keys_to_update:
        pts = [points[i]["pos"] for i in key]
        poly = Polygon([(p.x, p.y) for p in pts])
        triangle_cache[key] = (poly, poly.bounds)

def bounding_boxes_intersect(bounds1, bounds2):
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2
    return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)

def draw_scene():
    screen.fill(WHITE)
    alpha = max(10, int(TRANSPARENT_BLUE_BASE_ALPHA / r))
    alpha = int(TRANSPARENT_BLUE_BASE_ALPHA / r)
    color = (0, 0, 255, alpha)    # Otherwise use blue

    for tri_keys in partition_triangles_keys:
        # Get polygons and bounding boxes
        polys_bounds = [triangle_cache[key] for key in tri_keys]

        # Quick bounding box intersection test before polygon intersection
        combined_bounds = polys_bounds[0][1]
        intersect_possible = True
        for _, b in polys_bounds[1:]:
            if not bounding_boxes_intersect(combined_bounds, b):
                intersect_possible = False
                break
            # update combined bounding box to intersection of previous bounds (optional)
            combined_bounds = (
                max(combined_bounds[0], b[0]),
                max(combined_bounds[1], b[1]),
                min(combined_bounds[2], b[2]),
                min(combined_bounds[3], b[3]),
            )
            if combined_bounds[0] > combined_bounds[2] or combined_bounds[1] > combined_bounds[3]:
                intersect_possible = False
                break
        if not intersect_possible:
            # Still draw triangles even if no intersection
            for poly, _ in polys_bounds:
                pygame.draw.polygon(screen, GREEN, list(poly.exterior.coords), 1)
            continue

        # Draw triangles
        for poly, _ in polys_bounds:
            pygame.draw.polygon(screen, GREEN, list(poly.exterior.coords), 1)

        # Compute intersection only if bounding boxes intersect
        intersect_poly = polys_bounds[0][0]
        for poly, _ in polys_bounds[1:]:
            intersect_poly = intersect_poly.intersection(poly)
            if intersect_poly.is_empty:
                break

        if intersect_poly and not intersect_poly.is_empty:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            if intersect_poly.geom_type == 'Polygon':
                pygame.draw.polygon(overlay, color, list(intersect_poly.exterior.coords))
            elif intersect_poly.geom_type == 'MultiPolygon':
                for part in intersect_poly.geoms:
                    pygame.draw.polygon(overlay, color, list(part.exterior.coords))
            screen.blit(overlay, (0, 0))

    # Draw points last
    for pt in points:
        pygame.draw.circle(screen, RED, (int(pt["pos"].x), int(pt["pos"].y)), RADIUS)
        label = font.render(pt["label"], True, BLACK)
        screen.blit(label, (pt["pos"].x + 10, pt["pos"].y - 10))

    pygame.display.flip()

dragging_point_index = None
drag_offset = pygame.Vector2(0, 0)
needs_redraw = True

running = True
while running:
    event_happened = False
    for event in pygame.event.get():
        event_happened = True

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.Vector2(event.pos)
                for i, pt in enumerate(points):
                    if pt["pos"].distance_to(mouse_pos) <= RADIUS:
                        dragging_point_index = i
                        drag_offset = pt["pos"] - mouse_pos
                        break

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging_point_index = None

        elif event.type == pygame.MOUSEMOTION:
            if dragging_point_index is not None:
                mouse_pos = pygame.Vector2(event.pos)
                new_pos = mouse_pos + drag_offset
                new_pos.x = max(RADIUS, min(WIDTH - RADIUS, new_pos.x))
                new_pos.y = max(RADIUS, min(HEIGHT - RADIUS, new_pos.y))
                points[dragging_point_index]["pos"] = new_pos

                # Update only affected triangles
                update_triangles_for_point(dragging_point_index)
                needs_redraw = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                points = generate_random_points()
                build_triangle_cache_and_partitions()
                needs_redraw = True
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_RIGHT:
                r += 1
                update_partitions(r)
                points = generate_random_points()
                build_triangle_cache_and_partitions()
                needs_redraw = True
            elif event.key == pygame.K_LEFT:
                if r > 1:
                    r -= 1
                    update_partitions(r)
                    points = generate_random_points()
                    build_triangle_cache_and_partitions()
                    needs_redraw = True

    if needs_redraw and event_happened:
        draw_scene()
        needs_redraw = False

    clock.tick(60)

pygame.quit()
sys.exit()
