import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.spatial import ConvexHull
import trimesh

def generate_random_points(r):
    """
    Generates 4*r + 1 random points in the unit cube.

    Args:
        r (int): The number of points to generate per dimension.

    Returns:
        np.ndarray: A 2D array of shape (4*r + 1, 3) containing the random points.
    """
    points = np.random.rand(4*r + 1, 3)
    print(points)
    return points

def plot_points(ax, points):
    """
    Plots the given points in 3D space and draws lines between all pairs of points.

    Args:
        ax (Axes3D): The axes to plot on.
        points (np.ndarray): A 2D array of shape (n, 3) containing the points to plot.
    """
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='k')
    
    # Draw lines between all pairs of points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            ax.plot3D(*zip(points[i], points[j]), c='g', alpha=0.2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

def iterate_divisions(points, r):
    """
    Iterates over all divisions of 4*r + 1 points into r-1 tetrahedra and one set of 5 points.

    Args:
        points (np.ndarray): A 2D array of shape (4*r + 1, 3) containing the points to divide.
        r (int): The number of points to generate per dimension.

    Yields:
        tuple: A tuple containing the points for each tetrahedron and the remaining 5 points.
    """
    num_points = len(points)
    num_tetrahedra = r - 1
    num_points_per_tetrahedron = 4

    # Generate all combinations of 4 points for each tetrahedron
    for tetrahedra_points in combinations(range(num_points), num_tetrahedra * num_points_per_tetrahedron):
        tetrahedra_points = np.array(tetrahedra_points)
        tetrahedra_points = tetrahedra_points.reshape(num_tetrahedra, num_points_per_tetrahedron)

        # Get the remaining 5 points
        remaining_points = [i for i in range(num_points) if i not in tetrahedra_points.flatten()]
        remaining_points = np.array(remaining_points)

        # Check if the remaining points are exactly 5
        if len(remaining_points)!= 5:
            continue

        # Yield the points for each tetrahedron and the remaining 5 points
        yield tetrahedra_points, remaining_points

def merge_hulls(hulls):
    if (len(hulls) <= 1): return hulls

    for j, hull in enumerate(hulls[1:]):
        hullU = hull.union(hulls[0])
        if (hullU.is_volume):
            hulls[j] = hullU
            return merge_hulls(hulls[1:])
        hullI = hull.intersection(hulls[0])
        if (hullI.is_volume):
            print(f"huh?? vol={hullI.volume}")
    
    return [hulls[0]] + merge_hulls(hulls[1:])

def draw(ax, points, r):
    ax.clear()
    ax.set_axis_off()
    ax.grid(False)
    plot_points(ax, points)
    hulls = []
    num_hulls = 0
    for i, (tetrahedra_points, remaining_points) in enumerate(iterate_divisions(points, r)):
        # print(f"Division {i+1}:")
        # print("Tetrahedra points:")
        # for j, tetrahedron_points in enumerate(tetrahedra_points):
        #     print(f"Tetrahedron {j+1}: {tetrahedron_points}")
        # print(f"Remaining points: {remaining_points}")
        # print()
        hull1 = trimesh.convex.convex_hull([[points[j] for j in l] for l in tetrahedra_points][0])
        hull2 = trimesh.convex.convex_hull([points[j] for j in remaining_points])
        hull12 = hull1.intersection(hull2)
        if hull12.is_volume:
            if hulls == []:
                hulls = [hull12]
                print("hull set!")
            else:
                # print(len(hulls))
                num_hulls += 1
                need_append = True
                for i, hull in enumerate(hulls):
                    hullp = hull.union(hull12)
                    if (hullp.is_volume):
                        hulls[i] = hullp
                        need_append = False
                        break
                
                if need_append:
                    hulls.append(hull12)
        # hulls = [hull1, hull2, hull12]
        # for hull in hulls:
        #     rmesh = trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces)
        #     ax.plot_trisurf(rmesh.vertices[:, 0], rmesh.vertices[:, 1], rmesh.vertices[:, 2], triangles=rmesh.faces, alpha=0.1)
        #     for face in hull.faces:
        #         face_points = hull.vertices[face]
        #         ax.plot3D(*zip(face_points[1], face_points[2]), c='black', linewidth=2.0)
        #         ax.plot3D(*zip(face_points[2], face_points[0]), c='black', linewidth=2.0)
        #         ax.plot3D(*zip(face_points[0], face_points[1]), c='black', linewidth=2.0)
        # rmesh = trimesh.Trimesh(vertices=hull12.vertices, faces=hull12.faces)
        # ax.plot_trisurf(rmesh.vertices[:, 0], rmesh.vertices[:, 1], rmesh.vertices[:, 2], triangles=rmesh.faces, alpha=0.8)
        # for face in hull12.faces:
        #     face_points = hull.vertices[face]
        #     ax.plot3D(*zip(face_points[1], face_points[2]), c='black', linewidth=2.0)
        #     ax.plot3D(*zip(face_points[2], face_points[0]), c='black', linewidth=2.0)
        #     ax.plot3D(*zip(face_points[0], face_points[1]), c='black', linewidth=2.0)
        # break
    print(f"Unioned {num_hulls} intersections")
    print(len(hulls))
    # hulls = merge_hulls(hulls)
    print(len(hulls))
    for hull in hulls:
        rmesh = trimesh.Trimesh(vertices=hull.vertices, faces=hull.faces)
        ax.plot_trisurf(rmesh.vertices[:, 0], rmesh.vertices[:, 1], rmesh.vertices[:, 2], triangles=rmesh.faces, alpha=0.1)
        for face in hull.faces:
            face_points = hull.vertices[face]
            ax.plot3D(*zip(face_points[1], face_points[2]), c='black', linewidth=2.0)
            ax.plot3D(*zip(face_points[2], face_points[0]), c='black', linewidth=2.0)
            ax.plot3D(*zip(face_points[0], face_points[1]), c='black', linewidth=2.0)

def on_key_press(event):
    global points, r, ax
    if event.key == 'r':
        points = generate_random_points(r)
        draw(ax, points, r)

def main():
    global points, r, ax
    r = 2  # Change this value to generate more or fewer points
    points = generate_random_points(r)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    draw(ax, points, r)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()


if __name__ == '__main__':
    main()