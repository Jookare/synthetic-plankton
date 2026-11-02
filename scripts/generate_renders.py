"""Render paired grayscale synthetic plankton images and ground-truth depth in Blender 5.2."""

import csv
import fnmatch
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.dont_write_bytecode = True


def write_png(path, pixels, bits):
    """Write data pixels directly, without Blender's display transform."""
    pixels = np.flipud(pixels)
    height, width = pixels.shape[:2]
    color = 0 if pixels.ndim == 2 else 2
    data = pixels.astype(">u2" if bits == 16 else "u1").tobytes()
    stride = width * (1 if color == 0 else 3) * (bits // 8)

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    raw = b"".join(b"\0" + data[i : i + stride] for i in range(0, len(data), stride))
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bits, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b""),
    )


# Build a fresh scene once at the start of every batch.
def setup(args):
    """Build the scene from launcher arguments and return scene/material/Z node.

    The material is a diffuse grey surface. The world uses Is Camera Ray so
    its grey background is visible without providing environmental fill light.
    Blender 5.2 requires a standalone compositor group and Group Output node.
    """
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.render.samples_per_pixel
    scene.cycles.use_denoising = args.render.denoise
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x, scene.render.resolution_y = args.render.width_px, args.render.height_px
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    # Camera calibration and target size follow data_generation_final.txt.
    bpy.ops.object.camera_add(location=(0, 0, args.camera.distance_units))
    scene.camera = bpy.context.object
    cam = scene.camera.data
    cam.lens, cam.sensor_width, cam.sensor_fit = args.camera.focal_length_mm, args.camera.sensor_width_mm, "HORIZONTAL"
    cam.clip_start, cam.clip_end = 0.000001, 1

    # Coaxial headlight: rays travel along the camera's viewing axis.
    # Disable cast shadows; diffuse surface orientation still gives shape cues.
    bpy.ops.object.light_add(type="SUN")
    light = bpy.context.object
    light.name = "CameraIllumination"
    light.location = scene.camera.location
    light.rotation_euler = scene.camera.rotation_euler
    light.data.energy = args.render.sun_strength
    light.data.angle = 0
    light.data.use_shadow = False

    world = bpy.data.worlds.new("MicroscopyBackground")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix = nt.nodes.new("ShaderNodeMixShader")
    lp = nt.nodes.new("ShaderNodeLightPath")
    dark = nt.nodes.new("ShaderNodeBackground")
    dark.inputs["Color"].default_value = (0, 0, 0, 1)
    grey = nt.nodes.new("ShaderNodeBackground")
    linear = (
        args.render.background_intensity / 12.92 if args.render.background_intensity <= 0.04045 else ((args.render.background_intensity + 0.055) / 1.055) ** 2.4
    )
    grey.inputs["Color"].default_value = (linear, linear, linear, 1)
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(dark.outputs[0], mix.inputs[1])
    nt.links.new(grey.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])

    mat = bpy.data.materials.new("PlanktonGrey")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    diff = mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (args.particle.diffuse_grey, args.particle.diffuse_grey, args.particle.diffuse_grey, 1)
    out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(diff.outputs[0], out.inputs["Surface"])

    # Preserve raw positive camera distances in EXR; normalize later in Python.
    scene.view_layers[0].use_pass_z = True
    tree = bpy.data.node_groups.new("PairedOutputs", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    layers = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], out.inputs["Image"])
    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.name = "DepthOutput"
    depth.file_output_items.clear()
    depth.file_output_items.new("FLOAT", "Depth")
    depth.format.media_type = "IMAGE"
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_depth = "32"
    depth.save_as_render = False
    tree.links.new(layers.outputs["Depth"], depth.inputs[0])
    return scene, mat, depth


def configure_device(scene, args):
    """Enable available Cycles hardware; AUTO falls back to CPU, explicit GPU fails."""
    scene.cycles.device = "CPU"
    if args.render.device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"] if args.render.device == "AUTO" else [args.render.device]:
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devices = [d for d in prefs.devices if d.type == backend]
                if devices:
                    for d in prefs.devices:
                        d.use = d in devices
                    scene.cycles.device = "GPU"
                    break
            except (TypeError, RuntimeError):
                continue
        if scene.cycles.device == "CPU" and args.render.device != "AUTO":
            msg = f"No available {args.render.device} device"
            raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Orientation sampling: Provides views from all sides of the object
# ---------------------------------------------------------------------------
def uniform_orientations(count, seed):
    """Fibonacci directions and seeded overall rotation."""
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    overall = Quaternion(
        (
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
        ),
    )
    rotations = []
    for i in range(count):
        z = 1 - 2 * (i + 0.5) / count
        theta = 2 * math.pi * (i + 0.5) / ((1 + math.sqrt(5)) / 2)
        direction = Vector((math.sqrt(1 - z * z) * math.cos(theta), math.sqrt(1 - z * z) * math.sin(theta), z))
        rotations.append(overall @ Vector((1, 0, 0)).rotation_difference(direction))
    return rotations


# ---------------------------------------------------------------------------
# Output processing: EXR preserves scene-unit distances; only image intensity gets noise.
# ---------------------------------------------------------------------------
def save_depth_outputs(destination, stem, args, scene):
    """Rename compositor EXR, normalize foreground depth and write its mask.

    EXR background is a large sentinel, not a valid surface distance. Excluding
    it before min/max normalization avoids destroying the foreground contrast.
    Reserve zero for background and use 1..65535 for foreground PNG values.
    """
    candidates = list(destination.glob(f"{stem}_raw_depth*.exr"))
    if len(candidates) != 1:
        msg = f"Expected one depth EXR, got {candidates}"
        raise RuntimeError(msg)
    
    raw = destination / f"{stem}_depth.exr"
    candidates[0].replace(raw)
    image = bpy.data.images.load(str(raw), check_existing=False)
    values = np.array(image.pixels[:], dtype=np.float32).reshape(args.render.height_px, args.render.width_px, image.channels)[:, :, 0]
    valid = np.isfinite(values) & (values > 0) & (values < scene.camera.data.clip_end)
    if not valid.any():
        msg = f"Empty depth map: {raw}"
        raise RuntimeError(msg)
    near, far = float(values[valid].min()), float(values[valid].max())
    normalized = np.zeros_like(values)
    normalized[valid] = (values[valid] - near) / (far - near) if far > near else 0
    if args.render.depth_convention == "near-white":
        normalized[valid] = 1 - normalized[valid]
    encoded = np.zeros(values.shape, dtype=np.uint16)
    encoded[valid] = 1 + np.rint(normalized[valid] * 65534).astype(np.uint16)
    write_png(destination / f"{stem}_depth.png", encoded, 16)
    write_png(destination / f"{stem}_mask.png", valid.astype(np.uint8) * 255, 8)
    bpy.data.images.remove(image)
    return near, far


def save_grayscale_image(path, args, rng):
    """Write single-channel intensity, optionally adding zero-mean Gaussian noise.

    This is sensor-like intensity noise, not independent colored RGB noise.
    Even with noise disabled, write one 8-bit grayscale channel. Depth is never
    passed through this function and receives no augmentation.
    """
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    pixels = np.array(image.pixels[:]).reshape(args.render.height_px, args.render.width_px, image.channels)
    intensity = pixels[:, :, :3].mean(axis=2)
    # PNG samples are already display-encoded; do not encode sRGB a second time.
    if args.render.noise_std:
        intensity = intensity + rng.normal(0, args.render.noise_std, intensity.shape)
    write_png(path, np.rint(np.clip(intensity, 0, 1) * 255), 8)
    bpy.data.images.remove(image)


def save_inspection_scene(destination, scene, depth, meshes, args):
    """Save a copy of the first posed scene, without turning it into a template.

    Keep the model, camera, material, light and compositor available for manual
    inspection. Manual F12 output has its own directory. Noise is a later Python
    processing step, so F12 shows the clean underlying render.
    """
    original = scene.render.filepath, depth.directory, depth.file_name
    preview = destination / "inspection.blend"
    try:
        scene.render.filepath = "//manual_preview/image.png"
        depth.directory = "//manual_preview"
        depth.file_name = "depth"
        scene["inspection_notes"] = "First sampled orientation. F12 writes manual_preview; Gaussian noise is applied only by the batch Python pipeline."
        scene["generation_arguments"] = json.dumps(args.settings)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        meshes[0].select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.spaces.active.clip_start = 0.000001
                    area.spaces.active.clip_end = 10
                    area.spaces.active.region_3d.view_perspective = "CAMERA"
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(preview), copy=True, relative_remap=False)
    finally:
        scene.render.filepath, depth.directory, depth.file_name = original
    return preview.name


# Per-model orchestration: import -> normalize -> orient -> render.
def render_model(path, destination, args, scene, mat, depth):
    """Render one source model; clean up only the objects imported here.

    Imported parts share a parent at the origin so each quaternion rotates the
    entire organism. Saved quaternions and normalization describe the pose
    applied to the original source geometry.
    """
    destination.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    before = set(bpy.data.objects)
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.wm.stl_import(filepath=str(path))
    imported = set(bpy.data.objects) - before
    meshes = sorted((o for o in imported if o.type == "MESH"), key=lambda o: o.name)
    if not meshes:
        msg = f"No meshes in {path}"
        raise RuntimeError(msg)
    bpy.context.view_layer.update()
    
    # Bounding-box center and one uniform scale preserve the source proportions.
    points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(p[i] for p in points) for i in range(3)])
    hi = Vector([max(p[i] for p in points) for i in range(3)])
    center = (lo + hi) / 2
    largest = max(hi - lo)
    if largest <= 0:
        msg = f"Zero-size model: {path}"
        raise RuntimeError(msg)
    target = args.particle.max_dimension_units
    if target <= 0:
        raise RuntimeError("target-size must be positive")
    scale = target / largest
    root = bpy.data.objects.new("Orientation", None)
    scene.collection.objects.link(root)
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)
    for obj in meshes:
        matrix = transform @ obj.matrix_world
        obj.parent = root
        obj.matrix_world = matrix
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
            poly.use_smooth = True
        obj.data.set_sharp_from_angle(angle=math.radians(60))
    metadata = []
    rng = np.random.default_rng(args.render.seed)
    for i, q in enumerate(uniform_orientations(args.render.views_per_model, args.render.seed)):
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = q
        scene.cycles.seed = (args.render.seed + i) % 2147483647
        stem = f"{i:04d}"
        scene.frame_set(i + 1)
        scene.render.filepath = str(destination / f"{stem}_rgb.png")
        depth.directory = str(destination)
        depth.file_name = f"{stem}_raw_depth"
        if i == 0 and args.save_scene:
            save_inspection_scene(destination, scene, depth, meshes, args)
        bpy.ops.render.render(write_still=True)
        # One render supplies aligned intensity/Z. Encoding/noise happen afterward.
        near, far = save_depth_outputs(destination, stem, args, scene)
        save_grayscale_image(scene.render.filepath, args, rng)
        metadata.append({"view": i, "quaternion_wxyz": list(q), "depth_min": near, "depth_max": far})
    # Record the source path, settings and orientations for reproduction.
    record = {
        "source": str(path),
        "inspection_scene": "inspection.blend" if args.save_scene else None,
        "geometry_coordinates": "Blender XYZ, Z-up, normalized, unrotated",
        "scale": scale,
        "center": list(center),
        "settings": args.settings,
        "effective_scene": {
            "samples": scene.cycles.samples,
            "denoising": scene.cycles.use_denoising,
            "width": args.render.width_px,
            "height": args.render.height_px,
            "target_size": target,
            "camera_matrix": [list(row) for row in scene.camera.matrix_world],
            "lens": scene.camera.data.lens,
            "sensor_width": scene.camera.data.sensor_width,
        },
        "orientation_mode": "uniform",
        "image_encoding": "8-bit single-channel grayscale",
        "raw_depth_units": "normalized Blender scene units, not calibrated real-world units",
        "views": metadata,
    }
    (destination / "metadata.json").write_text(json.dumps(record, indent=2))
    with (destination / "rotations.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_w", "q_x", "q_y", "q_z"])
        writer.writerows(v["quaternion_wxyz"] for v in metadata)
        
    # Keep the camera, material and lights for the next organism.
    for obj in [*list(imported), root]:
        data = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            bpy.data.meshes.remove(data)


def main(settings=None):
    """Read launcher settings, build a fresh scene, then iterate source models."""
    if settings is None:
        settings = json.loads(sys.argv[sys.argv.index("--") + 1])

    # Mirror the YAML sections directly, without another argparse or YAML loader.
    args = SimpleNamespace(
        input=Path(settings["input_dir"]),
        output=Path(settings["output_dir"]),
        render=SimpleNamespace(**settings["render"]),
        camera=SimpleNamespace(**settings["camera"]),
        particle=SimpleNamespace(**settings["particle"]),
        model_pattern=settings.get("model_pattern", "*"),
        save_scene=settings.get("save_scene", False),
        settings=settings,
    )

    # Do some initial validation of the settings before starting Blender rendering.
    if args.render.views_per_model <= 0 or any(
        v is not None and v <= 0 for v in (args.render.samples_per_pixel, args.render.width_px, args.render.height_px, args.render.max_models)
    ):
        raise ValueError("Counts and dimensions must be positive")
    if not math.isfinite(args.render.noise_std) or args.render.noise_std < 0:
        raise ValueError("Noise must be finite and nonnegative")
    if args.render.seed < 0:
        raise ValueError("seed must be nonnegative")
    positive = (args.camera.distance_units, args.camera.focal_length_mm, args.camera.sensor_width_mm, args.particle.max_dimension_units)
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        raise ValueError("Camera dimensions and target-size must be finite and positive")
    if (
        not 0 <= args.render.background_intensity <= 1
        or not 0 <= args.particle.diffuse_grey <= 1
        or not math.isfinite(args.render.sun_strength)
        or args.render.sun_strength < 0
    ):
        raise ValueError("Background/object-grey must be 0..1 and sun-energy must be nonnegative")

    # All input/output paths are relative to the repository root, not the current working directory.
    args.input, args.output = args.input.resolve(), args.output.resolve()

    # Load model files
    supported_extensions = {".obj", ".stl"}

    models = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in supported_extensions and fnmatch.fnmatchcase(p.stem, args.model_pattern))
    if not models:
        msg = f"No matching 3D models found in {args.input}"
        raise RuntimeError(msg)

    scene, mat, depth = setup(args)
    configure_device(scene, args)
    for path in models[: args.render.max_models]:
        destination = args.output / path.relative_to(args.input).with_suffix("")
        render_model(path, destination, args, scene, mat, depth)


if __name__ == "__main__":
    main()
