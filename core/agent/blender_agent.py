from enum import Enum
import math
import random
import os
import sys
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import traceback
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / conditional Blender import
# ---------------------------------------------------------------------------
try:
    import bpy
    import bmesh

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    bmesh = None
    BLENDER_AVAILABLE = False
else:
    if os.environ.get("EDIATH_ALLOW_BLENDER", "0") == "1":
        try:
            import atexit
            import addon_utils

            def _safe_disable_addons():
                try:
                    mods = list(getattr(addon_utils, "modules", lambda: [])())
                    for m in mods:
                        try:
                            name = getattr(m, "__name__", None)
                            if not name:
                                continue
                            try:
                                enabled, _, _ = addon_utils.check(name)
                            except Exception:
                                enabled = False
                            if enabled:
                                try:
                                    addon_utils.disable(name)
                                except Exception:
                                    pass
                        except Exception:
                            continue
                except Exception:
                    pass

            try:
                atexit.register(_safe_disable_addons)
            except Exception:
                pass
        except Exception:
            pass
    else:
        logger.debug("Skipping Blender addon cleanup (EDIATH_ALLOW_BLENDER not set)")


def _blender_ge(major: int, minor: int = 0) -> bool:
    if not BLENDER_AVAILABLE:
        return False
    return bpy.app.version >= (major, minor, 0)


def _get_blender_version() -> str:
    if not BLENDER_AVAILABLE:
        return "not_installed"
    return ".".join(str(v) for v in bpy.app.version)


# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------


class RenderEngine(Enum):
    CYCLES = "CYCLES"
    EEVEE = "BLENDER_EEVEE"
    WORKBENCH = "BLENDER_WORKBENCH"


class ExportFormat(Enum):
    FBX = "fbx"
    OBJ = "obj"
    STL = "stl"
    GLTF = "gltf"
    USD = "usd"
    ABC = "abc"
    PLY = "ply"


class MaterialType(Enum):
    DEFAULT = "default"
    METAL = "metal"
    GLASS = "glass"
    EMISSIVE = "emissive"
    TRANSPARENT = "transparent"
    SUBSURFACE = "subsurface"
    SKIN = "skin"
    FUR = "fur"
    SCALE = "scale"


@dataclass
class RenderSettings:
    engine: RenderEngine = RenderEngine.CYCLES
    samples: int = 64
    resolution_x: int = 1920
    resolution_y: int = 1080
    use_denoising: bool = True
    use_motion_blur: bool = False
    film_transparent: bool = False
    output_format: str = "PNG"


@dataclass
class AnimationKeyframe:
    frame: int
    location: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None
    scale: Optional[Tuple[float, float, float]] = None


@dataclass
class VertexGroup:
    """Represents a group of vertices for mesh manipulation"""

    name: str
    vertices: List[int]
    weight: float = 1.0


@dataclass
class MeshPart:
    """Represents a part of a complex mesh (for animal creation)"""

    name: str
    shape_type: str  # sphere, cube, cylinder, cone, custom
    location: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    color: Optional[Tuple[float, float, float, float]] = None
    parent: Optional[str] = None


# ---------------------------------------------------------------------------
# Enhanced Blender 3D Agent
# ---------------------------------------------------------------------------


class Blender3DAgent:
    """Enhanced 3D scene builder capable of creating complex organic shapes"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.render_settings = RenderSettings()
        self._error_count = 0
        self._warning_count = 0
        self._created_objects: List[Any] = []
        self._collections: Dict[str, Any] = {}
        self._vertex_groups: Dict[str, VertexGroup] = {}
        self._blender_available = (
            BLENDER_AVAILABLE and os.environ.get("EDIATH_ALLOW_BLENDER", "0") == "1"
        )

        if not self._blender_available:
            logger.warning("[BlenderAgent] bpy not available — running in stub mode")
            return

        self._clear_scene()
        self.setup_scene()
        self._setup_from_config()
        logger.info("[BlenderAgent] Initialized | Blender %s", _get_blender_version())

    def _check_blender(self) -> None:
        if not self._blender_available:
            raise RuntimeError("Blender (bpy) is not available.")

    def _setup_from_config(self):
        self._check_blender()
        if "render" in self.config:
            render_config = self.config["render"]
            self.render_settings.engine = RenderEngine(
                render_config.get("engine", "CYCLES")
            )
            self.render_settings.samples = render_config.get("samples", 64)
            self.render_settings.resolution_x = render_config.get("resolution_x", 1920)
            self.render_settings.resolution_y = render_config.get("resolution_y", 1080)
            self.render_settings.use_denoising = render_config.get(
                "use_denoising", True
            )
            self._apply_render_settings()

    def _clear_scene(self) -> None:
        self._check_blender()
        try:
            objects_to_delete = [
                o
                for o in bpy.data.objects
                if o.type in {"MESH", "CURVE", "FONT", "LIGHT", "CAMERA", "EMPTY"}
                and o.name not in {"Camera", "Light"}
            ]
            if objects_to_delete:
                bpy.ops.object.select_all(action="DESELECT")
                for obj in objects_to_delete:
                    obj.select_set(True)
                bpy.ops.object.delete(use_global=False)
        except Exception as e:
            logger.exception("Scene clear failed: %s", e)

    def setup_scene(self) -> None:
        """Set up camera, lights, and world"""
        self._check_blender()

        def get_active():
            try:
                return bpy.context.view_layer.objects.active
            except Exception:
                return None

        # Camera
        if len(bpy.data.cameras) == 0:
            try:
                bpy.ops.object.camera_add(location=(7, -7, 5))
                camera = get_active()
                if camera:
                    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
                    camera.name = "MainCamera"
            except Exception as e:
                logger.exception("Camera add failed: %s", e)

        # Lighting
        try:
            # Sun light
            if len(bpy.data.lights) == 0:
                bpy.ops.object.light_add(type="SUN", location=(5, -5, 10))
                sun = get_active()
                if sun:
                    sun.data.energy = 3
                    sun.name = "SunLight"

            # Fill light
            bpy.ops.object.light_add(type="POINT", location=(0, 0, 5))
            fill = get_active()
            if fill:
                fill.data.energy = 1
                fill.name = "FillLight"

            # Rim light
            bpy.ops.object.light_add(type="POINT", location=(-3, 3, 4))
            rim = get_active()
            if rim:
                rim.data.energy = 1.5
                rim.name = "RimLight"
        except Exception as e:
            logger.exception("Light setup failed: %s", e)

        # World background
        try:
            world = bpy.context.scene.world
            if world:
                world.use_nodes = True
                nodes = world.node_tree.nodes
                links = world.node_tree.links
                nodes.clear()
                bg_node = nodes.new(type="ShaderNodeBackground")
                bg_node.inputs[0].default_value = (0.05, 0.05, 0.1, 1.0)
                bg_node.inputs[1].default_value = 1.0
                output_node = nodes.new(type="ShaderNodeOutputWorld")
                links.new(bg_node.outputs[0], output_node.inputs[0])
        except Exception as e:
            logger.exception("World setup failed: %s", e)

        self._apply_render_settings()

    def _apply_render_settings(self):
        self._check_blender()
        scene = bpy.context.scene
        scene.render.engine = self.render_settings.engine.value
        scene.render.resolution_x = self.render_settings.resolution_x
        scene.render.resolution_y = self.render_settings.resolution_y
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = self.render_settings.output_format

        if self.render_settings.engine == RenderEngine.CYCLES:
            scene.cycles.samples = self.render_settings.samples
            scene.cycles.use_denoising = self.render_settings.use_denoising
            scene.cycles.device = "GPU" if self.config.get("use_gpu", False) else "CPU"
        elif self.render_settings.engine == RenderEngine.EEVEE:
            scene.eevee.taa_render_samples = self.render_settings.samples

    # ---------------------------------------------------------------------------
    # Advanced Mesh Manipulation Methods
    # ---------------------------------------------------------------------------

    def create_mesh_from_vertices(
        self,
        vertices: List[Tuple],
        faces: List[List[int]],
        name: str = "CustomMesh",
        location: Tuple = (0, 0, 0),
    ) -> Any:
        """Create a mesh from custom vertex and face data"""
        self._check_blender()

        mesh_data = bpy.data.meshes.new(name + "_mesh")
        mesh_data.from_pydata(vertices, [], faces)
        mesh_data.update()

        obj = bpy.data.objects.new(name, mesh_data)
        bpy.context.collection.objects.link(obj)
        obj.location = location

        self._created_objects.append(obj)
        return obj

    def extrude_face(self, obj: Any, face_index: int, amount: float = 0.5) -> Any:
        """Extrude a specific face of a mesh"""
        self._check_blender()

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        if face_index < len(bm.faces):
            face = bm.faces[face_index]
            result = bmesh.ops.extrude_face_region(bm, geom=[face])
            verts = [v for v in result["geom"] if isinstance(v, bmesh.types.BMVert)]
            bmesh.ops.translate(bm, verts=verts, vec=(0, 0, amount))

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode="OBJECT")

        return obj

    def subdivide_mesh(self, obj: Any, subdivisions: int = 2) -> Any:
        """Apply subdivision surface modifier for smooth organic shapes"""
        self._check_blender()

        modifier = obj.modifiers.new(name="Subdivision", type="SUBSURF")
        if _blender_ge(4, 0):
            modifier.levels = subdivisions
            modifier.render_levels = subdivisions
        else:
            modifier.levels = subdivisions

        return obj

    def add_mirror_modifier(self, obj: Any, axis: str = "X") -> Any:
        """Add mirror modifier for symmetrical modeling"""
        self._check_blender()

        modifier = obj.modifiers.new(name="Mirror", type="MIRROR")
        if axis == "X":
            modifier.use_axis_x = True
        elif axis == "Y":
            modifier.use_axis_y = True
        elif axis == "Z":
            modifier.use_axis_z = True

        return obj

    def add_lattice_deform(
        self, obj: Any, resolution: Tuple[int, int, int] = (3, 3, 3)
    ) -> Any:
        """Add lattice deform modifier for organic shape manipulation"""
        self._check_blender()

        # Create lattice
        bpy.ops.object.add(type="LATTICE")
        lattice = bpy.context.active_object
        lattice.name = f"{obj.name}_Lattice"
        lattice.location = obj.location

        # Set lattice resolution
        lattice.data.points_u = resolution[0]
        lattice.data.points_v = resolution[1]
        lattice.data.points_w = resolution[2]

        # Add lattice modifier to object
        modifier = obj.modifiers.new(name="Lattice", type="LATTICE")
        modifier.object = lattice

        self._created_objects.append(lattice)
        return lattice

    def sculpt_smooth(self, obj: Any, iterations: int = 10) -> Any:
        """Apply smooth shading and mesh smoothing"""
        self._check_blender()

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        for _ in range(iterations):
            bpy.ops.mesh.vertices_smooth()
        bpy.ops.object.mode_set(mode="OBJECT")

        return obj

    # ---------------------------------------------------------------------------
    # Shape Composition System (For Animals)
    # ---------------------------------------------------------------------------

    def create_animal_body(
        self,
        body_type: str = "quadruped",
        size: Tuple[float, float, float] = (1.0, 0.5, 0.8),
        location: Tuple = (0, 0, 0),
    ) -> Dict[str, Any]:
        """Create a basic animal body structure"""
        self._check_blender()

        parts = {}

        if body_type == "quadruped":
            # Main body (scaled cube)
            body = self.create_cube(
                location=(location[0], location[1], location[2] + 0.3),
                size=1.0,
                name="Body",
            )
            body.scale = size
            bpy.ops.object.transform_apply(scale=True)
            self.set_object_smooth(body)
            parts["body"] = body

            # Legs
            leg_positions = [
                (-0.4, 0.3, 0),
                (0.4, 0.3, 0),
                (-0.4, -0.3, 0),
                (0.4, -0.3, 0),
            ]
            for i, pos in enumerate(leg_positions):
                leg = self.create_cylinder(
                    location=(
                        location[0] + pos[0],
                        location[1] + pos[1],
                        location[2] - 0.2,
                    ),
                    radius=0.15,
                    depth=0.5,
                    name=f"Leg_{i}",
                )
                self.set_object_smooth(leg)
                parts[f"leg_{i}"] = leg

            # Head
            head = self.create_sphere(
                location=(location[0], location[1] + 0.5, location[2] + 0.5),
                radius=0.3,
                name="Head",
            )
            self.set_object_smooth(head)
            parts["head"] = head

            # Tail
            tail = self.create_cone(
                location=(location[0], location[1] - 0.5, location[2] + 0.2),
                radius=0.1,
                depth=0.4,
                name="Tail",
            )
            parts["tail"] = tail

        elif body_type == "biped":
            # Torso
            torso = self.create_cube(
                location=(location[0], location[1], location[2] + 0.5),
                size=1.0,
                name="Torso",
            )
            torso.scale = (0.6, 0.4, 1.0)
            bpy.ops.object.transform_apply(scale=True)
            self.set_object_smooth(torso)
            parts["torso"] = torso

            # Head
            head = self.create_sphere(
                location=(location[0], location[1], location[2] + 1.0),
                radius=0.35,
                name="Head",
            )
            self.set_object_smooth(head)
            parts["head"] = head

            # Arms
            for side, x_offset in [("left", -0.5), ("right", 0.5)]:
                arm = self.create_cylinder(
                    location=(location[0] + x_offset, location[1], location[2] + 0.7),
                    radius=0.1,
                    depth=0.8,
                    name=f"Arm_{side}",
                )
                self.set_object_smooth(arm)
                parts[f"arm_{side}"] = arm

            # Legs
            for side, x_offset in [("left", -0.3), ("right", 0.3)]:
                leg = self.create_cylinder(
                    location=(location[0] + x_offset, location[1], location[2] - 0.2),
                    radius=0.12,
                    depth=0.7,
                    name=f"Leg_{side}",
                )
                self.set_object_smooth(leg)
                parts[f"leg_{side}"] = leg

        return parts

    def create_lion(self, location: Tuple = (0, 0, 0)) -> Dict[str, Any]:
        """Create a stylized lion using primitive composition"""
        self._check_blender()

        parts = {}

        # Main body (ellipsoid)
        body = self.create_sphere(
            location=(location[0], location[1], location[2] + 0.4),
            radius=0.5,
            name="Lion_Body",
        )
        body.scale = (0.9, 0.6, 1.2)
        bpy.ops.object.transform_apply(scale=True)
        self.add_material(body, (0.8, 0.5, 0.2, 1.0))  # Golden brown
        self.set_object_smooth(body)
        parts["body"] = body

        # Head
        head = self.create_sphere(
            location=(location[0], location[1] + 0.7, location[2] + 0.35),
            radius=0.32,
            name="Lion_Head",
        )
        self.add_material(head, (0.75, 0.45, 0.15, 1.0))
        self.set_object_smooth(head)
        parts["head"] = head

        # Mane (torus scaled)
        mane = self.create_torus(
            location=(location[0], location[1] + 0.68, location[2] + 0.35),
            major_radius=0.45,
            minor_radius=0.08,
            name="Lion_Mane",
        )
        mane.scale = (1.0, 0.8, 0.7)
        self.add_material(mane, (0.6, 0.3, 0.1, 1.0), material_type=MaterialType.FUR)
        parts["mane"] = mane

        # Ears
        ear_positions = [(-0.25, 0.75, 0.55), (0.25, 0.75, 0.55)]
        for i, pos in enumerate(ear_positions):
            ear = self.create_cone(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] + pos[2],
                ),
                radius=0.12,
                depth=0.15,
                name=f"Lion_Ear_{i}",
            )
            self.add_material(ear, (0.7, 0.4, 0.15, 1.0))
            parts[f"ear_{i}"] = ear

        # Legs
        leg_positions = [
            (-0.4, -0.3, 0.1),
            (0.4, -0.3, 0.1),
            (-0.4, 0.3, 0.1),
            (0.4, 0.3, 0.1),
        ]
        for i, pos in enumerate(leg_positions):
            leg = self.create_cylinder(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] - 0.2,
                ),
                radius=0.12,
                depth=0.55,
                name=f"Lion_Leg_{i}",
            )
            self.add_material(leg, (0.7, 0.45, 0.2, 1.0))
            self.set_object_smooth(leg)
            parts[f"leg_{i}"] = leg

        # Tail
        tail = self.create_cylinder(
            location=(location[0], location[1] - 0.65, location[2] + 0.2),
            radius=0.07,
            depth=0.5,
            name="Lion_Tail",
        )
        self.add_material(tail, (0.7, 0.45, 0.2, 1.0))

        tail_tip = self.create_sphere(
            location=(location[0], location[1] - 0.9, location[2] + 0.25),
            radius=0.09,
            name="Lion_TailTip",
        )
        self.add_material(tail_tip, (0.5, 0.25, 0.1, 1.0))
        parts["tail"] = tail
        parts["tail_tip"] = tail_tip

        # Nose
        nose = self.create_sphere(
            location=(location[0], location[1] + 0.88, location[2] + 0.38),
            radius=0.07,
            name="Lion_Nose",
        )
        self.add_material(nose, (0.2, 0.1, 0.05, 1.0))
        parts["nose"] = nose

        # Eyes
        eye_positions = [(-0.12, 0.85, 0.48), (0.12, 0.85, 0.48)]
        for i, pos in enumerate(eye_positions):
            eye = self.create_sphere(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] + pos[2],
                ),
                radius=0.05,
                name=f"Lion_Eye_{i}",
            )
            self.add_material(eye, (0.1, 0.1, 0.1, 1.0))
            parts[f"eye_{i}"] = eye

            # Eye highlight
            highlight = self.create_sphere(
                location=(
                    location[0] + pos[0] + 0.02,
                    location[1] + pos[1] + 0.02,
                    location[2] + pos[2] + 0.03,
                ),
                radius=0.02,
                name=f"Lion_Highlight_{i}",
            )
            self.add_material(highlight, (1.0, 1.0, 1.0, 1.0))
            parts[f"highlight_{i}"] = highlight

        logger.info("Lion created successfully!")
        return parts

    def create_dog(self, location: Tuple = (0, 0, 0)) -> Dict[str, Any]:
        """Create a stylized dog"""
        self._check_blender()

        parts = {}

        # Body
        body = self.create_cube(
            location=(location[0], location[1], location[2] + 0.3),
            size=0.8,
            name="Dog_Body",
        )
        body.scale = (0.8, 0.5, 1.0)
        bpy.ops.object.transform_apply(scale=True)
        self.add_material(body, (0.6, 0.4, 0.2, 1.0))
        self.set_object_smooth(body)
        parts["body"] = body

        # Head
        head = self.create_sphere(
            location=(location[0], location[1] + 0.65, location[2] + 0.45),
            radius=0.28,
            name="Dog_Head",
        )
        self.add_material(head, (0.55, 0.35, 0.15, 1.0))
        self.set_object_smooth(head)
        parts["head"] = head

        # Ears (floppy)
        ear_positions = [(-0.22, 0.7, 0.62), (0.22, 0.7, 0.62)]
        for i, pos in enumerate(ear_positions):
            ear = self.create_cube(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] + pos[2],
                ),
                size=0.15,
                name=f"Dog_Ear_{i}",
            )
            ear.scale = (0.6, 0.3, 1.2)
            bpy.ops.object.transform_apply(scale=True)
            self.add_material(ear, (0.5, 0.3, 0.1, 1.0))
            parts[f"ear_{i}"] = ear

        # Legs
        leg_positions = [
            (-0.3, -0.25, 0.1),
            (0.3, -0.25, 0.1),
            (-0.3, 0.25, 0.1),
            (0.3, 0.25, 0.1),
        ]
        for i, pos in enumerate(leg_positions):
            leg = self.create_cylinder(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] - 0.15,
                ),
                radius=0.1,
                depth=0.45,
                name=f"Dog_Leg_{i}",
            )
            self.add_material(leg, (0.55, 0.35, 0.15, 1.0))
            parts[f"leg_{i}"] = leg

        # Tail (wagging)
        tail = self.create_cone(
            location=(location[0], location[1] - 0.55, location[2] + 0.25),
            radius=0.08,
            depth=0.35,
            name="Dog_Tail",
        )
        self.add_material(tail, (0.5, 0.3, 0.1, 1.0))
        parts["tail"] = tail

        # Nose
        nose = self.create_sphere(
            location=(location[0], location[1] + 0.82, location[2] + 0.48),
            radius=0.06,
            name="Dog_Nose",
        )
        self.add_material(nose, (0.15, 0.1, 0.05, 1.0))
        parts["nose"] = nose

        return parts

    def create_bird(
        self, location: Tuple = (0, 0, 0), species: str = "eagle"
    ) -> Dict[str, Any]:
        """Create a stylized bird"""
        self._check_blender()

        parts = {}

        # Body
        body = self.create_sphere(
            location=(location[0], location[1], location[2] + 0.3),
            radius=0.35,
            name=f"{species}_Body",
        )
        body.scale = (0.7, 0.5, 1.1)
        bpy.ops.object.transform_apply(scale=True)
        color = (0.4, 0.35, 0.35, 1.0) if species == "eagle" else (0.8, 0.7, 0.4, 1.0)
        self.add_material(body, color)
        self.set_object_smooth(body)
        parts["body"] = body

        # Head
        head = self.create_sphere(
            location=(location[0], location[1] + 0.55, location[2] + 0.5),
            radius=0.22,
            name=f"{species}_Head",
        )
        self.add_material(head, color)
        self.set_object_smooth(head)
        parts["head"] = head

        # Beak
        beak = self.create_cone(
            location=(location[0], location[1] + 0.72, location[2] + 0.55),
            radius=0.08,
            depth=0.18,
            name=f"{species}_Beak",
        )
        self.add_material(beak, (0.9, 0.7, 0.2, 1.0))
        parts["beak"] = beak

        # Wings
        wing_positions = [(-0.45, 0.1, 0.35), (0.45, 0.1, 0.35)]
        for i, pos in enumerate(wing_positions):
            wing = self.create_cube(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] + pos[2],
                ),
                size=0.4,
                name=f"{species}_Wing_{i}",
            )
            wing.scale = (0.3, 1.2, 0.6)
            bpy.ops.object.transform_apply(scale=True)
            self.add_material(wing, color)
            parts[f"wing_{i}"] = wing

        # Tail feathers
        tail = self.create_cone(
            location=(location[0], location[1] - 0.5, location[2] + 0.3),
            radius=0.12,
            depth=0.3,
            name=f"{species}_Tail",
        )
        self.add_material(tail, color)
        parts["tail"] = tail

        return parts

    def create_fish(
        self, location: Tuple = (0, 0, 0), fish_type: str = "goldfish"
    ) -> Dict[str, Any]:
        """Create a stylized fish"""
        self._check_blender()

        parts = {}

        # Body
        body = self.create_sphere(
            location=(location[0], location[1], location[2]),
            radius=0.4,
            name=f"{fish_type}_Body",
        )
        body.scale = (1.2, 0.6, 0.8)
        bpy.ops.object.transform_apply(scale=True)
        color = (
            (0.9, 0.6, 0.2, 1.0) if fish_type == "goldfish" else (0.3, 0.5, 0.8, 1.0)
        )
        self.add_material(body, color)
        self.set_object_smooth(body)
        parts["body"] = body

        # Tail fin
        tail = self.create_cone(
            location=(location[0] - 0.55, location[1], location[2]),
            radius=0.2,
            depth=0.35,
            name=f"{fish_type}_Tail",
        )
        tail.scale = (0.8, 1.2, 0.5)
        self.add_material(tail, color)
        parts["tail"] = tail

        # Dorsal fin
        dorsal = self.create_cone(
            location=(location[0], location[1], location[2] + 0.45),
            radius=0.15,
            depth=0.25,
            name=f"{fish_type}_Dorsal",
        )
        dorsal.scale = (0.6, 1.0, 0.8)
        self.add_material(dorsal, color)
        parts["dorsal"] = dorsal

        # Eyes
        eye_positions = [(0.35, -0.2, 0.2), (0.35, 0.2, 0.2)]
        for i, pos in enumerate(eye_positions):
            eye = self.create_sphere(
                location=(
                    location[0] + pos[0],
                    location[1] + pos[1],
                    location[2] + pos[2],
                ),
                radius=0.08,
                name=f"{fish_type}_Eye_{i}",
            )
            self.add_material(eye, (0.1, 0.1, 0.1, 1.0))
            parts[f"eye_{i}"] = eye

        return parts

    def create_snake(self, location: Tuple = (0, 0, 0), length: int = 5) -> List[Any]:
        """Create a segmented snake body"""
        self._check_blender()

        segments = []
        segment_size = 0.3
        start_x = location[0]

        for i in range(length):
            segment = self.create_sphere(
                location=(start_x + (i * segment_size * 0.8), location[1], location[2]),
                radius=0.2,
                name=f"Snake_Segment_{i}",
            )
            segment.scale = (0.9, 1.0, 0.8)
            self.add_material(segment, (0.3, 0.7, 0.2, 1.0))
            self.set_object_smooth(segment)
            segments.append(segment)

        # Head
        head = self.create_sphere(
            location=(
                start_x + (length * segment_size * 0.8) + 0.2,
                location[1],
                location[2],
            ),
            radius=0.25,
            name="Snake_Head",
        )
        self.add_material(head, (0.35, 0.75, 0.25, 1.0))
        self.set_object_smooth(head)
        segments.append(head)

        # Tongue
        tongue = self.create_cone(
            location=(
                start_x + (length * segment_size * 0.8) + 0.45,
                location[1] + 0.05,
                location[2],
            ),
            radius=0.03,
            depth=0.15,
            name="Snake_Tongue",
        )
        self.add_material(tongue, (0.9, 0.2, 0.2, 1.0))
        segments.append(tongue)

        return segments

    # ---------------------------------------------------------------------------
    # Material System Enhancements
    # ---------------------------------------------------------------------------

    def add_material(
        self,
        obj: Any,
        color: Optional[Tuple] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
        material_type: MaterialType = MaterialType.DEFAULT,
        emission_strength: float = 1.0,
        transmission: float = 0.0,
    ) -> Any:
        """Add material with enhanced types for animals"""
        self._check_blender()

        mat_name = f"Mat_{obj.name}"
        if color is None:
            color = self.random_color()

        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

        # Special material handling
        if material_type == MaterialType.SKIN:
            roughness = 0.4
            subsurface = 0.3
            bsdf.inputs["Subsurface"].default_value = subsurface
        elif material_type == MaterialType.FUR:
            roughness = 0.8
        elif material_type == MaterialType.SCALE:
            metallic = 0.6
            roughness = 0.2
        elif material_type == MaterialType.METAL:
            metallic = 0.9
            roughness = 0.3
        elif material_type == MaterialType.GLASS:
            transmission = 0.9
            roughness = 0.1
        elif material_type == MaterialType.EMISSIVE:
            emission_strength = 5.0
        elif material_type == MaterialType.TRANSPARENT:
            transmission = 0.8
            roughness = 0.2
        elif material_type == MaterialType.SUBSURFACE:
            bsdf.inputs["Subsurface"].default_value = 0.5
            bsdf.inputs["Subsurface Color"].default_value = color

        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Transmission"].default_value = transmission
        bsdf.inputs["Emission Strength"].default_value = emission_strength

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (300, 0)
        links.new(bsdf.outputs[0], output.inputs[0])

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        return mat

    @staticmethod
    def random_color() -> Tuple[float, float, float, float]:
        return (
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
            1.0,
        )

    @staticmethod
    def set_object_smooth(obj: Any, angle: float = 60.0) -> None:
        if not BLENDER_AVAILABLE or obj.type != "MESH":
            return
        try:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
            if hasattr(obj.data, "use_auto_smooth"):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(angle)
        except RuntimeError:
            pass

    # ---------------------------------------------------------------------------
    # Basic Primitive Methods (Keep from original)
    # ---------------------------------------------------------------------------

    def create_cube(
        self, location=(0, 0, 0), size=1.0, name="Cube", color=None, smooth=False
    ):
        self._check_blender()
        bpy.ops.mesh.primitive_cube_add(size=size, location=location)
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(obj, color or (0.4, 0.6, 0.8, 1.0))
        if smooth:
            self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    def create_sphere(
        self, location=(0, 0, 0), radius=1.0, subdivisions=32, name="Sphere", color=None
    ):
        self._check_blender()
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            segments=subdivisions,
            ring_count=subdivisions // 2,
            location=location,
        )
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(obj, color or (0.8, 0.3, 0.3, 1.0))
        self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    def create_cylinder(
        self,
        location=(0, 0, 0),
        radius=0.5,
        depth=2.0,
        vertices=32,
        name="Cylinder",
        color=None,
    ):
        self._check_blender()
        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=depth, vertices=vertices, location=location
        )
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(obj, color or (0.3, 0.8, 0.4, 1.0))
        self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    def create_cone(
        self,
        location=(0, 0, 0),
        radius=0.8,
        depth=2.0,
        vertices=32,
        name="Cone",
        color=None,
    ):
        self._check_blender()
        bpy.ops.mesh.primitive_cone_add(
            radius1=radius, depth=depth, vertices=vertices, location=location
        )
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(obj, color or (0.9, 0.7, 0.2, 1.0))
        self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    def create_torus(
        self,
        location=(0, 0, 0),
        major_radius=1.0,
        minor_radius=0.3,
        major_segments=48,
        minor_segments=24,
        name="Torus",
        color=None,
    ):
        self._check_blender()
        bpy.ops.mesh.primitive_torus_add(
            major_radius=major_radius,
            minor_radius=minor_radius,
            major_segments=major_segments,
            minor_segments=minor_segments,
            location=location,
        )
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(
            obj, color or (0.6, 0.3, 0.7, 1.0), metallic=0.5, roughness=0.3
        )
        self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    def create_plane(self, location=(0, 0, 0), size=10.0, name="Ground", color=None):
        self._check_blender()
        bpy.ops.mesh.primitive_plane_add(size=size, location=location)
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(
            obj, color or (0.2, 0.2, 0.3, 1.0), metallic=0.0, roughness=0.9
        )
        self._created_objects.append(obj)
        return obj

    def create_monkey(self, location=(0, 0, 0), name="Monkey", color=None):
        self._check_blender()
        bpy.ops.mesh.primitive_monkey_add(location=location)
        obj = bpy.context.active_object
        obj.name = name
        self.add_material(obj, color or (0.8, 0.6, 0.4, 1.0))
        self.set_object_smooth(obj)
        self._created_objects.append(obj)
        return obj

    # ---------------------------------------------------------------------------
    # Animation Methods
    # ---------------------------------------------------------------------------

    def animate_rotation(
        self,
        obj,
        frame_start=1,
        frame_end=120,
        rotation_z=360,
        rotation_x=0,
        rotation_y=0,
    ):
        self._check_blender()
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_start)
        obj.rotation_euler = (
            math.radians(rotation_x),
            math.radians(rotation_y),
            math.radians(rotation_z),
        )
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_end)

    def animate_location(
        self, obj, frame_start=1, frame_end=120, start_loc=None, end_loc=None
    ):
        self._check_blender()
        if start_loc is None:
            start_loc = obj.location.copy()
        if end_loc is None:
            end_loc = (start_loc[0], start_loc[1], start_loc[2] + 2)
        obj.location = start_loc
        obj.keyframe_insert(data_path="location", frame=frame_start)
        obj.location = end_loc
        obj.keyframe_insert(data_path="location", frame=frame_end)

    def animate_walk_cycle(self, objects: List[Any], frame_end: int = 60):
        """Simple walk cycle animation for legs"""
        for i, obj in enumerate(objects):
            phase = i * (math.pi * 2 / len(objects))
            for frame in range(0, frame_end + 1, 10):
                t = frame / frame_end
                offset = math.sin(t * math.pi * 2 + phase) * 0.1
                obj.location.z += offset
                obj.keyframe_insert(data_path="location", frame=frame)

    # ---------------------------------------------------------------------------
    # Export and Render
    # ---------------------------------------------------------------------------

    def export_scene(
        self, filepath: str, fmt: str = "fbx", use_selection: bool = False
    ) -> bool:
        self._check_blender()
        filepath = os.path.abspath(filepath)
        fmt = fmt.lower()
        try:
            if fmt == "fbx":
                bpy.ops.export_scene.fbx(filepath=filepath, use_selection=use_selection)
            elif fmt == "obj":
                if _blender_ge(4):
                    bpy.ops.wm.obj_export(
                        filepath=filepath, export_selected_objects=use_selection
                    )
                else:
                    bpy.ops.export_scene.obj(
                        filepath=filepath, use_selection=use_selection
                    )
            elif fmt == "gltf":
                bpy.ops.export_scene.gltf(
                    filepath=filepath, export_selected=use_selection
                )
            else:
                return False
            print(f"✅ Scene exported → {filepath}")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def render_image(self, filepath: str = "render.png") -> bool:
        self._check_blender()
        bpy.context.scene.render.filepath = os.path.abspath(filepath)
        try:
            bpy.ops.render.render(write_still=True)
            print(f"✅ Render saved → {filepath}")
            return True
        except Exception as e:
            print(f"Render failed: {e}")
            return False

    # ---------------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "blender_version": _get_blender_version(),
            "created_objects": len(self._created_objects),
            "mesh_objects": (
                len([o for o in bpy.data.objects if o.type == "MESH"])
                if BLENDER_AVAILABLE
                else 0
            ),
            "collections": len(self._collections),
            "errors": self._error_count,
            "warnings": self._warning_count,
        }

    def clear_scene(self) -> None:
        self._check_blender()
        self._clear_scene()
        self._created_objects.clear()
        self._collections.clear()


# ---------------------------------------------------------------------------
# EDIATH Wrapper
# ---------------------------------------------------------------------------


class BlenderAgent:
    """Wrapper for EDIATH integration"""

    def __init__(self, config: Optional[Dict] = None):
        self.agent: Optional[Blender3DAgent] = None
        self.agent_type = "blender_3d"
        self.capabilities = [
            "create_animal_lion",
            "create_animal_dog",
            "create_animal_bird",
            "create_animal_fish",
            "create_animal_snake",
            "create_animal_body",
            "subdivide_mesh",
            "extrude_face",
            "add_mirror",
            "sculpt_smooth",
            "create_primitive",
            "animate",
            "render",
            "export",
        ]
        self._initialized = False

        if BLENDER_AVAILABLE:
            try:
                self.agent = Blender3DAgent(config)
                self._initialized = True
            except Exception as e:
                print(f"[BlenderAgent] Init failed: {e}")

    async def initialize(self) -> bool:
        return self._initialized

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agent:
            return {"success": False, "error": "Blender not available"}

        operation = request.get("operation")
        params = request.get("params", {})

        try:
            if operation == "create_lion":
                parts = self.agent.create_lion(**params)
                return {
                    "success": True,
                    "parts": list(parts.keys()),
                    "message": "Lion created!",
                }

            elif operation == "create_dog":
                parts = self.agent.create_dog(**params)
                return {
                    "success": True,
                    "parts": list(parts.keys()),
                    "message": "Dog created!",
                }

            elif operation == "create_bird":
                parts = self.agent.create_bird(**params)
                return {
                    "success": True,
                    "parts": list(parts.keys()),
                    "message": "Bird created!",
                }

            elif operation == "create_fish":
                parts = self.agent.create_fish(**params)
                return {
                    "success": True,
                    "parts": list(parts.keys()),
                    "message": "Fish created!",
                }

            elif operation == "create_snake":
                parts = self.agent.create_snake(**params)
                return {
                    "success": True,
                    "segments": len(parts),
                    "message": "Snake created!",
                }

            elif operation == "create_animal_body":
                parts = self.agent.create_animal_body(**params)
                return {
                    "success": True,
                    "parts": list(parts.keys()),
                    "message": "Animal body created!",
                }

            elif operation == "subdivide":
                obj = bpy.data.objects.get(params.get("object_name"))
                if obj:
                    self.agent.subdivide_mesh(obj, params.get("subdivisions", 2))
                    return {"success": True, "message": "Mesh subdivided"}
                return {"success": False, "error": "Object not found"}

            elif operation == "extrude":
                obj = bpy.data.objects.get(params.get("object_name"))
                if obj:
                    self.agent.extrude_face(
                        obj, params.get("face_index", 0), params.get("amount", 0.5)
                    )
                    return {"success": True, "message": "Face extruded"}
                return {"success": False, "error": "Object not found"}

            elif operation == "mirror":
                obj = bpy.data.objects.get(params.get("object_name"))
                if obj:
                    self.agent.add_mirror_modifier(obj, params.get("axis", "X"))
                    return {"success": True, "message": "Mirror modifier added"}
                return {"success": False, "error": "Object not found"}

            elif operation == "smooth":
                obj = bpy.data.objects.get(params.get("object_name"))
                if obj:
                    self.agent.sculpt_smooth(obj, params.get("iterations", 10))
                    return {"success": True, "message": "Mesh smoothed"}
                return {"success": False, "error": "Object not found"}

            elif operation == "render":
                success = self.agent.render_image(params.get("filepath", "render.png"))
                return {
                    "success": success,
                    "message": "Render completed" if success else "Render failed",
                }

            elif operation == "export":
                success = self.agent.export_scene(
                    params.get("filepath", "output.fbx"), params.get("format", "fbx")
                )
                return {
                    "success": success,
                    "message": "Export completed" if success else "Export failed",
                }

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Blender3DAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "initialized": self._initialized,
        }

    async def close(self):
        self._initialized = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not BLENDER_AVAILABLE:
        print("Blender not available. Run inside Blender with EDIATH_ALLOW_BLENDER=1")
        sys.exit(0)

    print("=" * 60)
    print("  🦁 Enhanced Blender 3D Agent - Animal Creator")
    print("=" * 60)

    agent = Blender3DAgent()

    # Create a lion
    print("\n🦁 Creating Lion...")
    lion_parts = agent.create_lion(location=(0, 0, 0))
    print(f"   Created {len(lion_parts)} parts: {list(lion_parts.keys())}")

    # Create a dog at different location
    print("\n🐕 Creating Dog...")
    dog_parts = agent.create_dog(location=(3, 0, 0))
    print(f"   Created {len(dog_parts)} parts")

    # Create an eagle
    print("\n🦅 Creating Eagle...")
    eagle_parts = agent.create_bird(location=(-3, 0, 1.5), species="eagle")
    print(f"   Created {len(eagle_parts)} parts")

    # Create a fish
    print("\n🐟 Creating Fish...")
    fish_parts = agent.create_fish(location=(0, 3, 0), fish_type="goldfish")
    print(f"   Created {len(fish_parts)} parts")

    # Position camera
    camera = bpy.data.objects.get("MainCamera")
    if camera:
        camera.location = (6, -8, 5)
        camera.rotation_euler = (math.radians(65), 0, math.radians(45))

    print("\n✅ Animals created successfully!")
    print(f"📊 Stats: {agent.get_stats()}")
