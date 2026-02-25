"""
Scene utilities:
  1. Load valid grid positions from pre-computed JSON files.
  2. Merge material XML (car_factory_integrate_xml) with layout mesh paths.
  3. Optionally inject human / robot mesh instances into the scene XML.

Coordinate convention (Sionna):  X = right, Z = up, Y = forward.
Position tuple order = [world_x, world_z, world_y].

Grid index → world rule:
    world_x = xi * 2 + 1
    world_z = height   (constant)
    world_y = -1.5 + yi * (-2)     ← grows more negative as yi increases

Usage:
    python src/scene_utils.py   # writes scenes/factory_standard.xml
"""
import os, re, json
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    PROJECT_ROOT, SCENE_DIR, SCENE_LAYOUT_DIR,
    MATERIAL_XML_PATH, MERGED_SCENE_XML,
    GRID_COORDS_JSON, ROBOT_HEIGHT, HUMAN_HEIGHT,
    HUMAN_MESH, ROBOT_MESH,
    gi_to_world,
)


# ── Grid position helpers ─────────────────────────────────────────────────────

def load_valid_grid_indices(json_path: str = GRID_COORDS_JSON) -> np.ndarray:
    """
    Load kept grid indices [[xi, yi], ...] from the scene JSON.
    Returns int32 ndarray of shape (N, 2).
    """
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data["kept"], dtype=np.int32)


def load_invalid_grid_indices(json_path: str = GRID_COORDS_JSON) -> np.ndarray:
    """Load obstacle (removed) grid indices [[xi, yi], ...]."""
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data["removed"], dtype=np.int32)


def grid_indices_to_world(indices: np.ndarray,
                           z: float = ROBOT_HEIGHT) -> np.ndarray:
    """
    Convert integer grid indices [[xi, yi], ...] to Sionna world positions.

    Sionna position tuple = [world_x, world_z, world_y]:
        world_x = xi * 2 + 1
        world_z = z  (height, constant)
        world_y = -1.5 + yi * (-2)

    Returns float32 ndarray of shape (N, 3) with column order [x, z, y].
    """
    N = len(indices)
    world = np.zeros((N, 3), dtype=np.float32)
    world[:, 0] = indices[:, 0] * 2 + 1            # x
    world[:, 1] = z                                  # z (height)
    world[:, 2] = -1.5 + indices[:, 1] * (-2)       # y  (forward, negative)
    return world


def load_grid_positions(json_path: str = GRID_COORDS_JSON,
                        z: float = ROBOT_HEIGHT) -> np.ndarray:
    """
    Convenience: load valid grid positions as Sionna [x, z, y] world coords.
    Returns float32 ndarray (N, 3).
    """
    indices = load_valid_grid_indices(json_path)
    return grid_indices_to_world(indices, z)


# ── XML merging ───────────────────────────────────────────────────────────────

def _get_mesh_map(layout_dir: str) -> dict:
    """Return {stem_name: abs_path} for all PLY files in layout/meshes/."""
    meshes_dir = os.path.join(layout_dir, "meshes")
    return {
        os.path.splitext(f)[0]: os.path.join(meshes_dir, f)
        for f in os.listdir(meshes_dir) if f.endswith(".ply")
    }


def build_merged_scene_xml(
    layout_dir: str = SCENE_LAYOUT_DIR,
    material_xml: str = MATERIAL_XML_PATH,
    output_path: str = MERGED_SCENE_XML,
    human_positions: list = None,   # list of [xi,yi] grid indices
    robot_positions: list = None,   # list of [xi,yi] grid indices
) -> str:
    """
    Generate a Sionna-compatible scene XML:
    - Takes radio-material BSDF from material_xml.
    - Patches 'meshes/X.ply' → absolute paths from layout_dir/meshes/.
    - Optionally appends human_clean.ply / robot_clean.ply at the given
      grid indices (converted to world [x, z, y] coordinates).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh_map = _get_mesh_map(layout_dir)

    with open(material_xml) as f:
        content = f.read()

    # Patch relative mesh paths → absolute
    def replace_mesh(match):
        stem     = os.path.splitext(os.path.basename(match.group(1)))[0]
        abs_path = mesh_map.get(stem)
        return f'value="{abs_path}"' if abs_path else match.group(0)

    content = re.sub(r'value="(meshes/[^"]+\.ply)"', replace_mesh, content)

    # Sionna 1.2.1 process_xml requires every BSDF to have an id attribute.
    # Add sequential ids to all inline <bsdf ... > tags that lack one.
    _bsdf_counter = [0]
    def _add_bsdf_id(m):
        tag = m.group(0)
        if 'id=' in tag:
            return tag          # already has id
        _id = f'mat_inline_{_bsdf_counter[0]}'
        _bsdf_counter[0] += 1
        return tag.rstrip('>').rstrip() + f' id="{_id}">'
    content = re.sub(r'<bsdf\b[^>]*>', _add_bsdf_id, content)

    # Strip closing </scene> to append agent shapes
    content = content.rstrip()
    if content.endswith("</scene>"):
        content = content[:-len("</scene>")].rstrip()

    # Append human instances
    if human_positions and os.path.exists(HUMAN_MESH):
        for i, pos in enumerate(human_positions):
            xi, yi = int(pos[0]), int(pos[1])
            wx, wz, wy = gi_to_world(xi, yi, HUMAN_HEIGHT)
            content += f"""
    <shape type="ply" id="human_{i}" name="human_{i}">
        <string name="filename" value="{HUMAN_MESH}"/>
        <transform name="to_world">
            <translate x="{wx}" y="{wy}" z="{wz}"/>
        </transform>
        <bsdf type="radio-material" id="mat_human_{i}">
            <float name="thickness" value="0.2"/>
            <float name="relative_permittivity" value="40.0"/>
            <float name="conductivity" value="0.8"/>
            <vector name="color" value="0.8, 0.6, 0.4"/>
        </bsdf>
    </shape>"""

    # Append mobile robot instances
    if robot_positions and os.path.exists(ROBOT_MESH):
        for i, pos in enumerate(robot_positions):
            xi, yi = int(pos[0]), int(pos[1])
            wx, wz, wy = gi_to_world(xi, yi, ROBOT_HEIGHT)
            content += f"""
    <shape type="ply" id="robot_agent_{i}" name="robot_agent_{i}">
        <string name="filename" value="{ROBOT_MESH}"/>
        <transform name="to_world">
            <translate x="{wx}" y="{wy}" z="{wz}"/>
        </transform>
        <bsdf type="radio-material" id="mat_robot_{i}">
            <float name="thickness" value="0.1"/>
            <float name="relative_permittivity" value="1.0"/>
            <float name="conductivity" value="5.0e6"/>
            <vector name="color" value="0.2, 0.4, 0.8"/>
        </bsdf>
    </shape>"""

    content += "\n</scene>\n"

    with open(output_path, "w") as f:
        f.write(content)

    print(f"[scene_utils] Merged scene XML → {output_path}")
    return output_path


if __name__ == "__main__":
    build_merged_scene_xml()

    gp = load_grid_positions()
    print(f"[scene_utils] Valid positions: {len(gp)}")
    print(f"  world_x  range: [{gp[:,0].min():.1f}, {gp[:,0].max():.1f}] m")
    print(f"  world_z  (ht):  {gp[:,1].min():.2f} m  (robot height)")
    print(f"  world_y  range: [{gp[:,2].min():.1f}, {gp[:,2].max():.1f}] m")
    print(f"  Example gi(0,0) → {gi_to_world(0,0,0.75)}")
    print(f"  Example gi(39,39) → {gi_to_world(39,39,0.75)}")
