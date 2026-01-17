import open3d as o3d
import numpy as np

# ============================================================
# LEGO COLOR PALETTES
# ============================================================

LEGO_PETAL_COLORS = [
    (np.array([1.0, 0.6, 0.75]), np.array([0.9, 0.2, 0.5])),   # Pink
    (np.array([1.0, 0.85, 0.3]), np.array([1.0, 0.65, 0.0])), # Yellow
    (np.array([0.8, 0.9, 1.0]), np.array([0.2, 0.4, 0.9])),  # Blue
    (np.array([0.95, 0.95, 0.95]), np.array([0.8, 0.1, 0.1])), # White-red
    (np.array([0.85, 0.7, 1.0]), np.array([0.4, 0.1, 0.7]))  # Purple
]

# ============================================================
# UTILITIES
# ============================================================

def add_surface_noise(mesh, scale=0.05, strength=0.08):
    vertices = np.asarray(mesh.vertices)
    noise = np.random.normal(scale=scale, size=vertices.shape)
    vertices += strength * noise
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

def color_gradient(mesh, inner_color, outer_color):
    vertices = np.asarray(mesh.vertices)
    r = np.linalg.norm(vertices[:, :2], axis=1)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)
    colors = (1 - r[:, None]) * inner_color + r[:, None] * outer_color
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# ============================================================
# SINGLE LEGO FLOWER
# ============================================================

def create_textured_flower():
    meshes = []

    # ---------------- STEM ----------------
    stem_height = 120
    stem = o3d.geometry.TriangleMesh.create_cylinder(radius=4, height=stem_height)
    # Move stem so bottom is at Z=0
    stem.translate([0, 0, stem_height / 2])
    add_surface_noise(stem, scale=0.03, strength=0.04)
    stem.paint_uniform_color([0.1, 0.6, 0.15]) # Green
    meshes.append(stem)

    # ---------------- CENTER ----------------
    center = o3d.geometry.TriangleMesh.create_sphere(radius=9)
    center.translate([0, 0, stem_height])
    add_surface_noise(center, scale=0.1, strength=0.15)
    center.paint_uniform_color([1.0, 0.8, 0.1]) # Yellow/Orange
    meshes.append(center)

    # ---------------- PETALS ----------------
    num_petals = 12
    petal_len, petal_w, petal_t = 26, 9, 4
    inner, outer = LEGO_PETAL_COLORS[np.random.randint(len(LEGO_PETAL_COLORS))]

    for i in range(num_petals):
        angle = 2 * np.pi * i / num_petals

        # LEGO-style blocky petal
        petal = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
        # Center the box at origin before transforming
        petal.translate([-0.5, -0.5, -0.5])

        S = np.eye(4)
        S[0, 0] = petal_len
        S[1, 1] = petal_w
        S[2, 2] = petal_t
        petal.transform(S)

        color_gradient(petal, inner, outer)

        R = petal.get_rotation_matrix_from_xyz((np.pi / 10, 0, angle))
        petal.rotate(R, center=(0, 0, 0))

        x = np.cos(angle) * 18
        y = np.sin(angle) * 18
        petal.translate([x, y, stem_height])

        meshes.append(petal)

    # ---------------- MERGE ----------------
    flower = meshes[0]
    for m in meshes[1:]:
        flower += m

    flower.compute_vertex_normals()
    return flower

# ============================================================
# BOUQUET
# ============================================================

def create_bouquet(num_flowers=9):
    bouquet = None

    for _ in range(num_flowers):
        flower = create_textured_flower()

        # Scale
        s = np.random.uniform(0.85, 1.15)
        flower.scale(s, center=(0, 0, 0))

        # Rotate randomly
        R = flower.get_rotation_matrix_from_xyz((
            np.random.uniform(-0.4, 0.4),
            np.random.uniform(-0.4, 0.4),
            np.random.uniform(0, 2*np.pi)
        ))
        flower.rotate(R, center=(0, 0, 0))

        # Translate into a bunch
        r = np.random.uniform(0, 25)
        t = np.random.uniform(0, 2*np.pi)
        flower.translate([
            r * np.cos(t),
            r * np.sin(t),
            np.random.uniform(-8, 8) # Height variation
        ])

        bouquet = flower if bouquet is None else bouquet + flower

    # Lift bouquet slightly so it sits inside the vase, not through the floor
    # Minimum Z is likely negative due to random rotation, let's fix roughly
    bouquet.translate([0, 0, 10])
    
    bouquet.compute_vertex_normals()
    return bouquet

# ============================================================
# LEGO VASE GENERATOR
# ============================================================

def create_lego_vase(radius=40, height=75, brick_height=8):
    meshes = []
    
    # Colors
    c_white = [0.92, 0.92, 0.95] 
    c_blue  = [0.2, 0.4, 0.8]  
    
    circumference = 2 * np.pi * radius
    brick_width = 12  # How wide is one lego brick
    num_bricks = int(circumference / brick_width)
    
    num_layers = int(height / brick_height)
    
    for layer in range(num_layers):
        # Shift every other layer for the "brick wall" pattern
        angle_offset = (np.pi / num_bricks) if (layer % 2 == 1) else 0
        
        # Color logic: Top layer is Blue, rest are White
        color = c_blue if layer == num_layers - 1 else c_white
        
        for i in range(num_bricks):
            angle = (2 * np.pi * i / num_bricks) + angle_offset
            
            # 1. Create Brick (Box)
            # height-0.5 creates a tiny gap between layers for realism
            brick = o3d.geometry.TriangleMesh.create_box(width=brick_width, height=brick_height-0.5, depth=8)
            
            # Center the brick geometry
            brick.translate([-brick_width/2, -brick_height/2, -4])
            
            # 2. Position in circle
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            z = layer * brick_height + (brick_height/2) # Start from Z=0 upwards
            
            # 3. Rotate to face outward
            R = brick.get_rotation_matrix_from_xyz((0, 0, angle + np.pi/2))
            brick.rotate(R, center=(0,0,0))
            brick.translate([x, y, z])
            
            brick.paint_uniform_color(color)
            add_surface_noise(brick, scale=0.02, strength=0.03) # Plastic noise
            meshes.append(brick)
            
            # 4. Add Studs on the top Blue rim
            if layer == num_layers - 1:
                stud = o3d.geometry.TriangleMesh.create_cylinder(radius=2, height=2)
                stud.translate([0, 0, 1 + brick_height/2]) # Sit on top
                stud.paint_uniform_color(color)
                
                # Move to same spot as brick
                stud.rotate(R, center=(0,0,0))
                stud.translate([x, y, z])
                meshes.append(stud)

    # Base Cylinder (Floor of the vase)
    base = o3d.geometry.TriangleMesh.create_cylinder(radius=radius-3, height=2)
    base.translate([0, 0, 1])
    base.paint_uniform_color(c_white)
    meshes.append(base)

    # Merge all bricks
    vase = meshes[0]
    for m in meshes[1:]:
        vase += m
        
    return vase

# ============================================================
# MAIN EXECUTION
# ============================================================

print("Generating LEGO Bouquet...")
bouquet = create_bouquet(num_flowers=8)

print("Building LEGO Vase...")
vase = create_lego_vase(radius=38, height=80)

# Combine
final_model = bouquet + vase
final_model.compute_vertex_normals()

print("Saving lego_bouquet.ply...")
o3d.visualization.draw([final_model])
o3d.io.write_triangle_mesh("lego_bouquet.ply", final_model)