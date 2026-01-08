import open3d as o3d

import open3d as o3d
import numpy as np

def create_lego_flower():
    meshes = []

    # ----------------------
    # 1. Stem
    # ----------------------
    stem_height = 120
    stem = o3d.geometry.TriangleMesh.create_cylinder(radius=4.0, height=stem_height)
    stem.paint_uniform_color([0.1, 0.7, 0.1])  # green
    stem.translate([0, 0, stem_height / 2])
    meshes.append(stem)

    # ----------------------
    # 2. Flower center
    # ----------------------
    center_radius = 10
    center = o3d.geometry.TriangleMesh.create_sphere(radius=center_radius)
    center.paint_uniform_color([1.0, 0.9, 0.1])  # yellow
    center.translate([0, 0, stem_height])
    meshes.append(center)

    # ----------------------
    # 3. Petals
    # ----------------------
    num_petals = 8
    petal_length = 25
    petal_width = 10

    for i in range(num_petals):
        angle = 2 * np.pi * i / num_petals

        # Petal shape (flattened sphere)
        petal = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        # Petal shape (flattened sphere)
        petal = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        petal.paint_uniform_color([1.0, 0.4, 0.7])  # pink

        # Non-uniform scaling using a transform
        S = np.eye(4)
        S[0, 0] = petal_length   # X scale
        S[1, 1] = petal_width    # Y scale
        S[2, 2] = 5              # Z scale

        petal.transform(S)

        petal.paint_uniform_color([1.0, 0.4, 0.7])  # pink

        # Rotate petal around Z
        R = petal.get_rotation_matrix_from_xyz((0, 0, angle))
        petal.rotate(R, center=(0, 0, 0))

        # Move petal outward from center
        x = np.cos(angle) * (center_radius + petal_length / 2)
        y = np.sin(angle) * (center_radius + petal_length / 2)
        petal.translate([x, y, stem_height])

        meshes.append(petal)

    # ----------------------
    # Combine everything
    # ----------------------
    flower = meshes[0]
    for m in meshes[1:]:
        flower += m

    flower.compute_vertex_normals()
    return flower


# Use this in your Part 3 code
flower = create_lego_flower()
o3d.visualization.draw([flower])
o3d.io.write_triangle_mesh("lego_flower.ply", flower)