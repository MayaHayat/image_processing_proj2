import open3d as o3d
import numpy as np

LEGO_PETAL_COLORS = [
    (np.array([1.0, 0.6, 0.75]), np.array([0.9, 0.2, 0.5])),   # Pink
    (np.array([1.0, 0.85, 0.3]), np.array([1.0, 0.65, 0.0])), # Yellow
    (np.array([0.8, 0.9, 1.0]), np.array([0.2, 0.4, 0.9])),  # Blue
    (np.array([0.9, 0.9, 0.9]), np.array([0.7, 0.1, 0.1])),  # White → Red tip
    (np.array([0.85, 0.7, 1.0]), np.array([0.4, 0.1, 0.7]))  # Purple
]


# --- SECTION 1: SURFACE NOISE ---
# This function makes the 3D model look "organic" and less like perfect plastic.
# It moves each vertex (point) of the mesh slightly in a random direction.
def add_surface_noise(mesh, scale=0.05, strength=0.08):
    # Convert mesh vertices to a numpy array for math operations
    vertices = np.asarray(mesh.vertices)
    # Generate random numbers (Gaussian noise) for X, Y, and Z
    noise = np.random.normal(scale=scale, size=vertices.shape)
    # Add that noise to the existing vertex positions
    vertices += strength * noise
    # Update the mesh with the new "bumpy" coordinates
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Re-calculate how light hits the bumpy surface
    mesh.compute_vertex_normals()

# --- SECTION 2: COLOR GRADIENT ---
# This creates a color transition (e.g., light pink center to dark pink edges).
def color_gradient(mesh, inner_color, outer_color):
    vertices = np.asarray(mesh.vertices)
    # Calculate the radial distance (radius 'r') of each vertex from the center (0,0)
    # np.linalg.norm calculates sqrt(x^2 + y^2)
    r = np.linalg.norm(vertices[:, :2], axis=1)
    # Normalize the distance so it is a value between 0 (center) and 1 (edge)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)

    # Linearly interpolate between the two colors based on distance 'r'
    # colors = (weight1 * color1) + (weight2 * color2)
    colors = (1 - r[:, None]) * inner_color + r[:, None] * outer_color
    # Apply these colors to the vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# --- SECTION 3: THE FLOWER GENERATOR ---
def create_textured_flower():
    meshes = [] # List to hold all parts (stem, center, petals)

    # --- PART A: THE STEM ---
    stem_height = 120
    # Create a 3D cylinder
    stem = o3d.geometry.TriangleMesh.create_cylinder(radius=4, height=stem_height)
    # Move it so it stands on the ground (Z=0) instead of being centered at origin
    stem.translate([0, 0, stem_height / 2])
    # Add minor bumps to make it look like a real plant stem
    add_surface_noise(stem, scale=0.2, strength=0.15)
    # Paint it green
    stem.paint_uniform_color([0.1, 0.6, 0.15])
    meshes.append(stem)

    # --- PART B: THE FLOWER CENTER ---
    center = o3d.geometry.TriangleMesh.create_sphere(radius=10)
    # Move it to the very top of the stem
    center.translate([0, 0, stem_height])
    # Add heavy noise to simulate the rough texture of seeds/pollen
    add_surface_noise(center, scale=0.3, strength=0.4)

    # Create a speckled yellow/orange look by randomizing vertex colors
    vertices = np.asarray(center.vertices)
    colors = np.random.uniform(
        [0.85, 0.85, 0.05], # Minimum yellow-orange
        [1.0, 1.0, 0.15],   # Maximum yellow-orange
        size=vertices.shape
    )
    center.vertex_colors = o3d.utility.Vector3dVector(colors)
    # meshes.append(center)

    # --- PART C: THE PETALS ---
    num_petals = 12
    petal_len, petal_w, petal_t = 28, 10, 4 # Dimensions: Length, Width, Thickness

    for i in range(num_petals):
        # Calculate the angle for this specific petal (360 degrees / 12)
        angle = 2 * np.pi * i / num_petals

        # Start with a simple sphere, then "squash" it into a petal shape
        petal = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

        # Apply a scaling matrix to turn the sphere into a long, flat oval
        S = np.eye(4)
        S[0, 0] = petal_len # Stretch in X
        S[1, 1] = petal_w   # Flatten slightly in Y
        S[2, 2] = petal_t   # Flatten a lot in Z
        petal.transform(S)

        # Apply the "vein" texture and the pink gradient
        add_surface_noise(petal, scale=0.15, strength=0.2)
        inner, outer = LEGO_PETAL_COLORS[np.random.randint(len(LEGO_PETAL_COLORS))]

        color_gradient(
            petal,
            inner_color=inner,
            outer_color=outer
        )

        # color_gradient(
        #     petal,
        #     inner_color=np.array([1.0, 0.6, 0.75]), # Light Pink
        #     outer_color=np.array([0.9, 0.2, 0.5])  # Dark Pink
        # )

        # Create a rotation matrix: tilt up (pi/10) and rotate around the stem (angle)
        R = petal.get_rotation_matrix_from_xyz((np.pi / 10, 0, angle))
        # Rotate the petal around the center of the flower
        petal.rotate(R, center=(0, 0, 0))

        # Position the petal at the correct circle coordinate (x, y) at the top of the stem
        x = np.cos(angle) * 18
        y = np.sin(angle) * 18
        petal.translate([x, y, stem_height])

        meshes.append(petal)

    # --- PART D: COMBINING ---
    # Merge all individual shapes into one single 3D object
    flower = meshes[0]
    for m in meshes[1:]:
        flower += m

    # Ensure lighting works correctly across the joined surfaces
    flower.compute_vertex_normals()
    return flower



def create_bouquet(
    num_flowers=7,
    spread_radius=25,
    height_jitter=10,
    scale_range=(0.85, 1.15)
):
    bouquet = None

    for i in range(num_flowers):
        flower = create_textured_flower()

        # --- Random scale ---
        s = np.random.uniform(*scale_range)
        flower.scale(s, center=(0, 0, 0))

        # --- Random rotation ---
        R = flower.get_rotation_matrix_from_xyz((
            np.random.uniform(-0.4, 0.4),   # tilt forward/back
            np.random.uniform(-0.4, 0.4),   # tilt left/right
            np.random.uniform(0, 2*np.pi)   # rotate around stem
        ))
        flower.rotate(R, center=(0, 0, 0))

        # --- Random position (clustered like a bouquet) ---
        r = np.random.uniform(0, spread_radius)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-height_jitter, height_jitter)

        flower.translate([x, y, z])

        # --- Merge into bouquet ---
        if bouquet is None:
            bouquet = flower
        else:
            bouquet += flower

    bouquet.compute_vertex_normals()
    return bouquet


# --- SECTION 4: EXECUTION ---
# # Generate the model
# flower = create_textured_flower()
# # Open a window to view and rotate the flower
# o3d.visualization.draw([flower])
# # Save it as a .ply file (you can use this for your final report/submission!)
# o3d.io.write_triangle_mesh("lego_flower.ply", flower)


bouquet = create_bouquet(num_flowers=9)

o3d.visualization.draw([bouquet])
o3d.io.write_triangle_mesh("lego_bouquet.ply", bouquet)
