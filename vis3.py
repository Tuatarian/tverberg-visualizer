import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

r = 2
num_points = 4 * r  # Must be exactly 4*r to partition into r tetrahedra
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
points = None
partitions = []
partition_index = 0

def generate_points():
    points = []
    for i in range(num_points):
        for attempt in range(10):
            point = np.random.rand(3) * 4 - 2
            if len(points) == 0 or min(np.linalg.norm(point - p) for p in points) > 0.5:
                points.append(point)
                break
        else:
            points.append(np.random.rand(3) * 4 - 2)
    return np.array(points)

def is_valid_tetrahedron(points4):
    if len(points4) != 4:
        return False
    a, b, c, d = points4
    matrix = np.array([b-a, c-a, d-a])
    det = np.linalg.det(matrix)
    volume = abs(det) / 6
    return volume > 1e-4

def tetrahedron_to_mesh(points4):
    if not is_valid_tetrahedron(points4):
        return None
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]
    try:
        mesh = trimesh.Trimesh(vertices=points4, faces=faces, process=False)
        mesh.fix_normals()
        mesh.process()
        if mesh.is_watertight and mesh.is_volume and abs(mesh.volume) > 1e-6:
            return mesh
    except:
        pass
    return None

def draw_mesh(ax, mesh, color='blue', alpha=0.6):
    if mesh is None:
        return
    for face in mesh.faces:
        tri = mesh.vertices[face]
        poly = Poly3DCollection([tri], facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_collection3d(poly)

def draw_wireframe(ax, mesh, color='black', linewidth=1.0):
    if mesh is None:
        return
    for edge in mesh.edges_unique:
        line = mesh.vertices[edge]
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=linewidth)

def generate_tetrahedron_partitions(points, r):
    """Generate all ways to partition n=4*r points into exactly r tetrahedra"""
    n = len(points)
    if n != 4 * r:
        print(f"Warning: Cannot partition {n} points into {r} tetrahedra of 4 points each")
        return []
    
    indices = list(range(n))
    partitions = []
    
    def backtrack(remaining_indices, current_partition):
        # If we've assigned all points, check if we have exactly r tetrahedra
        if not remaining_indices:
            if len(current_partition) == r:
                # Verify all tetrahedra are valid
                if all(is_valid_tetrahedron(points[list(group)]) for group in current_partition):
                    partitions.append([list(group) for group in current_partition])
            return
        
        # If we already have r tetrahedra but still have points left, invalid
        if len(current_partition) == r:
            return
            
        # If we don't have enough points left to complete the remaining tetrahedra, invalid
        remaining_tetrahedra = r - len(current_partition)
        if len(remaining_indices) != remaining_tetrahedra * 4:
            return
        
        # Try all combinations of 4 points from remaining indices
        for group in combinations(remaining_indices, 4):
            if is_valid_tetrahedron(points[list(group)]):
                new_remaining = [idx for idx in remaining_indices if idx not in group]
                backtrack(new_remaining, current_partition + [group])
    
    print(f"Generating partitions of {n} points into {r} tetrahedra...")
    backtrack(indices, [])
    print(f"Found {len(partitions)} valid partitions")
    
    # Remove duplicate partitions (same partition with different ordering)
    unique_partitions = []
    for partition in partitions:
        # Sort each tetrahedron's indices and sort tetrahedra by their first index
        normalized = [sorted(tetra) for tetra in partition]
        normalized.sort()
        if normalized not in unique_partitions:
            unique_partitions.append(normalized)
    
    print(f"After removing duplicates: {len(unique_partitions)} unique partitions")
    return unique_partitions

def is_valid_volume_mesh(mesh):
    """Check if a mesh is a valid 3D volume for boolean operations"""
    if mesh is None or mesh.is_empty:
        return False
    try:
        return (mesh.is_watertight and 
                mesh.is_volume and 
                hasattr(mesh, 'volume') and 
                abs(mesh.volume) > 1e-10)
    except:
        return False

def safe_intersection(meshes):
    """Safely compute intersection of all meshes"""
    valid_meshes = [m for m in meshes if is_valid_volume_mesh(m)]
    
    if not valid_meshes:
        return None
    
    if len(valid_meshes) == 1:
        return valid_meshes[0]
    
    print(f"Computing intersection of {len(valid_meshes)} valid meshes...")
    
    try:
        # Start with the first mesh and intersect with each subsequent mesh
        intersection = valid_meshes[0]
        print(f"  Starting with mesh volume: {intersection.volume:.6f}")
        
        for i, mesh in enumerate(valid_meshes[1:], 1):
            print(f"  Intersecting with mesh {i+1}/{len(valid_meshes)} (volume: {mesh.volume:.6f})")
            new_intersection = intersection.intersection(mesh)
            
            if is_valid_volume_mesh(new_intersection):
                intersection = new_intersection
                print(f"    Result volume: {intersection.volume:.6f}")
            else:
                print(f"    Intersection became invalid or empty")
                return None
                
        return intersection
        
    except Exception as e:
        print(f"  Intersection failed: {e}")
        return None

def safe_union(meshes):
    """Safely compute union of meshes, filtering out invalid ones"""
    valid_meshes = [m for m in meshes if is_valid_volume_mesh(m)]
    
    if not valid_meshes:
        return None
    
    if len(valid_meshes) == 1:
        return valid_meshes[0]
    
    print(f"Attempting union of {len(valid_meshes)} valid meshes...")
    
    try:
        # Sequential union approach
        union_mesh = valid_meshes[0]
        print(f"  Starting with mesh volume: {union_mesh.volume:.6f}")
        
        for i, mesh in enumerate(valid_meshes[1:], 1):
            print(f"  Unioning with mesh {i+1}/{len(valid_meshes)} (volume: {mesh.volume:.6f})")
            new_union = union_mesh.union(mesh)
            
            # if is_valid_volume_mesh(new_union):
            if is_valid_volume_mesh(new_union):
                union_mesh = new_union
                print(f"    Result volume: {union_mesh.volume:.6f}")
            else:
                print(f"    Degenerate mesh {i+1}, skipping")
                continue
        
        print(f"Volume gap between Union and Hull: {union_mesh.convex_hull.volume - union_mesh.volume}")
        return union_mesh
        
    except Exception as e:
        print(f"  Union failed: {e}")
        return None

def plot_all_intersections(points, partitions):
    ax.cla()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o', s=50)
    
    # Draw lines between all points
    for i, j in combinations(range(len(points)), 2):
        xs = [points[i, 0], points[j, 0]]
        ys = [points[i, 1], points[j, 1]]
        zs = [points[i, 2], points[j, 2]]
        ax.plot(xs, ys, zs, color='green', linewidth=0.3, alpha=0.3)

    all_intersections = []
    
    # For each partition, compute the intersection of all r tetrahedra in that partition
    for partition_idx, partition in enumerate(partitions):
        print(f"\nProcessing partition {partition_idx + 1}/{len(partitions)}:")
        
        # Convert each tetrahedron in this partition to a mesh
        meshes = []
        for tetra_idx, tetra_indices in enumerate(partition):
            mesh = tetrahedron_to_mesh(points[tetra_indices])
            if mesh:
                meshes.append(mesh)
                print(f"  Tetrahedron {tetra_idx + 1}: volume = {mesh.volume:.6f}")
            else:
                print(f"  Tetrahedron {tetra_idx + 1}: invalid")
        
        # Compute intersection of all tetrahedra in this partition
        if len(meshes) == r:  # We should have exactly r valid tetrahedra
            intersection = safe_intersection(meshes)
            if intersection:
                all_intersections.append(intersection)
                print(f"  Partition intersection volume: {intersection.volume:.6f}")
            else:
                print(f"  Partition intersection: empty or invalid")
        else:
            print(f"  Partition has only {len(meshes)}/{r} valid tetrahedra")

    # Draw all intersections with the same color
    intersection_color = 'blue'
    for inter in all_intersections:
        draw_mesh(ax, inter, color=intersection_color, alpha=0.6)

    # Compute and draw union wireframe
    if all_intersections:
        print(f"\nFound {len(all_intersections)} partition intersections")
        union_mesh = safe_union(all_intersections)
        if union_mesh:
            print(f"Union of all intersections: volume = {union_mesh.volume:.6f}")
            draw_wireframe(ax, union_mesh, color='black', linewidth=2.0)
        else:
            print("Union failed - drawing individual wireframes")
            for inter in all_intersections:
                draw_wireframe(ax, inter, color='black', linewidth=1.0)
    else:
        ax.text2D(0.5, 0.5, "No intersections found", transform=ax.transAxes, 
                  ha='center', va='center', fontsize=12, color='red')

    # Set plot limits and appearance
    all_coords = points.flatten()
    margin = 0.5
    ax.axis('off')
    ax.set_xlim([all_coords.min() - margin, all_coords.max() + margin])
    ax.set_ylim([all_coords.min() - margin, all_coords.max() + margin])
    ax.set_zlim([all_coords.min() - margin, all_coords.max() + margin])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title(f'Intersections of {r}-Tetrahedron Partitions ({len(all_intersections)} found)')
    plt.draw()
    return len(all_intersections)

def regenerate():
    global points, partitions, partition_index, num_points
    num_points = 4 * r  # Ensure we have exactly 4*r points
    points = generate_points()
    partitions = generate_tetrahedron_partitions(points, r)
    partition_index = 0
    plot_all_intersections(points, partitions)

def on_key(event):
    global r, num_points
    if event.key == 'r':
        regenerate()
    elif event.key in ['+', '-']:
        zoom_step = 0.1 if event.key == '+' else -0.1
        def zoom(get, set):
            lim = get()
            mid = (lim[0] + lim[1]) / 2
            span = (lim[1] - lim[0]) * (1 - zoom_step)
            set([mid - span / 2, mid + span / 2])
        zoom(ax.get_xlim3d, ax.set_xlim3d)
        zoom(ax.get_ylim3d, ax.set_ylim3d)
        zoom(ax.get_zlim3d, ax.set_zlim3d)
        plt.draw()
    elif event.key == 'right':
        r += 1
        regenerate()
    elif event.key == 'left':
        if r > 1:
            r -= 1
            regenerate()

fig.canvas.mpl_connect('key_press_event', on_key)
print("Controls:")
print("  'r' - Regenerate points and partitions")
print("  '+' - Zoom in")
print("  '-' - Zoom out")
print("  'right' - Increase r (more tetrahedra per partition)")
print("  'left' - Decrease r (fewer tetrahedra per partition)")
print(f"\nStarting with r={r} (partitions of {4*r} points into {r} tetrahedra)")
regenerate()
plt.show()