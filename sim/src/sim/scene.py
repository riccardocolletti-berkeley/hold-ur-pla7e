"""MuJoCo scene builder: UR5e arm + rectangular plate + ball + floor.

The stock UR5e model from MuJoCo Menagerie is loaded, augmented with a plate
on ``wrist_3_link``, a free ball on top, a floor and a light, then re-loaded
into a fresh `MjModel`. Intermediate and composed XMLs are written to a
per-process temp path so multiple `SubprocVecEnv` workers do not clobber
one another.

Plate size, ball mass, and friction coefficients come from `SceneConfig`.
The only side-effect on the project tree is an optional clone of MuJoCo
Menagerie into ``sim/external/mujoco_menagerie/`` (gitignored).
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

#: Sim package root: ``ball_on_plate/sim/``. Computed from this file's path so
#: behaviour is independent of the current working directory.
_SIM_ROOT = Path(__file__).resolve().parents[2]

#: Where external dependencies live. The Menagerie clone is the only one for now.
_EXTERNAL_DIR = _SIM_ROOT / "external"

#: Local clone of the MuJoCo Menagerie. Cloned on first use, gitignored.
_MENAGERIE_DIR = _EXTERNAL_DIR / "mujoco_menagerie"
_MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
_UR5E_XML = _MENAGERIE_DIR / "universal_robots_ur5e" / "ur5e.xml"

#: Optional decal image overlaid on the plate (Berkeley logo + course name).
#: If absent, the plate is rendered in plain wood colour.
DECAL_PATH = Path(__file__).resolve().parent / "assets" / "plate_decal.png"


@dataclass(frozen=True)
class SceneConfig:
    """Physical parameters consumed by `build_scene`. SI units throughout."""

    plate_size: tuple[float, float]  # full edge lengths along plate (x, y) [m]
    plate_thickness: float  # plate slab full height [m]
    plate_mass: float  # [kg]
    plate_friction: float  # sliding-friction coefficient
    ball_radius: float  # [m]
    ball_mass: float  # [kg]
    ball_friction: float  # sliding-friction coefficient
    # Adapter pose in tool0 frame. Both callers fill these from
    # ``ballplate.hardware.AdapterSpec`` so the simulator and the real-robot
    # static TF read the same numbers: no fallback default.
    adapter_position: tuple[float, float, float]
    adapter_orientation: tuple[float, float, float]


def build_scene(cfg: SceneConfig) -> mujoco.MjModel:
    """Build the simulation scene and return a freshly-compiled `MjModel`."""
    _ensure_menagerie()

    # Load the stock UR5e and dump its resolved XML so we can splice the
    # plate, ball, and lighting in. Per-PID file names guarantee that a
    # `SubprocVecEnv` of N workers does not race on shared paths.
    base_model = mujoco.MjModel.from_xml_path(str(_UR5E_XML))
    pid = os.getpid()
    resolved_xml = f"/tmp/ur5e_resolved_{pid}.xml"
    mujoco.mj_saveLastXML(resolved_xml, base_model)

    tree = ET.parse(resolved_xml)
    root = tree.getroot()

    # The composed XML lives under /tmp/, so any relative `meshdir` /
    # `texturedir` set by the menagerie's compiler tag must be rewritten to
    # absolute paths before re-loading or `from_xml_path` cannot resolve the
    # `.obj` meshes that ship with the UR5e model.
    _absolutize_asset_dirs(root, base_dir=_UR5E_XML.parent)
    _set_offscreen_framebuffer(root)

    worldbody = root.find("worldbody")
    wrist3 = _find_body(worldbody, "wrist_3_link")
    if wrist3 is None:
        raise RuntimeError("Could not find wrist_3_link in UR5e XML.")

    tool0 = _make_tool0_body(wrist3)
    _attach_riser(tool0, cfg)
    _attach_plate(tool0, root, cfg)
    _attach_ball(worldbody, cfg)
    _ensure_light(worldbody)
    _ensure_floor(worldbody)

    # The composed XML is written under /tmp (not into the menagerie clone)
    # so the external repo on disk stays pristine and reproducible.
    composed_xml = f"/tmp/ur5e_plate_ball_{pid}.xml"
    tree.write(composed_xml)
    return mujoco.MjModel.from_xml_path(composed_xml)


# ============================================================================
# Internals
# ============================================================================


def _ensure_menagerie() -> None:
    """Clone Google DeepMind's MuJoCo Menagerie into `sim/external/` if missing."""
    if _MENAGERIE_DIR.exists():
        return
    _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", _MENAGERIE_REPO, str(_MENAGERIE_DIR)],
        check=True,
    )


def _absolutize_asset_dirs(root: ET.Element, base_dir: Path) -> None:
    """Rewrite relative `meshdir`/`texturedir` on `<compiler>` to absolute paths.

    `mj_saveLastXML` preserves the menagerie's relative ``meshdir="assets/"``,
    which only resolves when the XML is loaded from the menagerie folder.
    The composed XML lives in /tmp, so we anchor every asset directory to
    `base_dir` (the directory holding the source `ur5e.xml`).
    """
    for compiler in root.iter("compiler"):
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not os.path.isabs(value):
                compiler.set(attr, str((base_dir / value).resolve()))


def _set_offscreen_framebuffer(root: ET.Element) -> None:
    """Bump the offscreen buffer so video recording works at 720p."""
    visual = root.find("visual") or ET.SubElement(root, "visual")
    gl = visual.find("global") or ET.SubElement(visual, "global")
    gl.set("offwidth", "1280")
    gl.set("offheight", "720")


def _find_body(elem: ET.Element, name: str):
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for child in elem:
        result = _find_body(child, name)
        if result is not None:
            return result
    return None


def _make_tool0_body(wrist3: ET.Element) -> ET.Element:
    """Mirror the Menagerie ``attachment_site`` as a body so the plate parents
    onto the UR ``tool0`` flange: the same frame the ROS launch's static TF
    uses. Without this indirection ``adapter_position`` and ``_orientation``
    are interpreted in the wrist_3_link cylinder frame, which is rotated 90°
    relative to ``tool0`` and offset 10 cm behind the flange.
    """
    site = wrist3.find("site[@name='attachment_site']")
    if site is None:
        raise RuntimeError(
            "wrist_3_link is missing the 'attachment_site' site that defines "
            "the UR tool0 flange. Re-clone the Menagerie."
        )
    tool0 = ET.SubElement(wrist3, "body")
    tool0.set("name", "tool0")
    tool0.set("pos", site.get("pos", "0 0 0"))
    tool0.set("quat", site.get("quat", "1 0 0 0"))
    return tool0


def _attach_riser(tool0: ET.Element, cfg: SceneConfig) -> None:
    """Visual-only cylinder from tool0 to the plate centre. No collision."""
    riser = ET.SubElement(tool0, "geom")
    riser.set("type", "cylinder")
    riser.set("fromto", f"0 0 0 0 0 {cfg.adapter_position[2]}")
    riser.set("size", "0.025")
    riser.set("rgba", "0.3 0.3 0.3 1")
    riser.set("contype", "0")
    riser.set("conaffinity", "0")


def _attach_plate(tool0: ET.Element, root: ET.Element, cfg: SceneConfig) -> None:
    """Bolt the plate onto ``tool0`` using ``hardware.yaml::adapter`` directly."""
    half_x = cfg.plate_size[0] / 2.0
    half_y = cfg.plate_size[1] / 2.0
    half_thick = cfg.plate_thickness / 2.0

    plate = ET.SubElement(tool0, "body")
    plate.set("name", "plate")
    plate.set("pos", " ".join(f"{x}" for x in cfg.adapter_position))
    plate.set("euler", " ".join(f"{x}" for x in cfg.adapter_orientation))

    plate_geom = ET.SubElement(plate, "geom")
    plate_geom.set("type", "box")
    plate_geom.set("size", f"{half_x} {half_y} {half_thick}")
    plate_geom.set("mass", str(cfg.plate_mass))
    plate_geom.set("rgba", "0.86 0.72 0.50 1")
    plate_geom.set("friction", f"{cfg.plate_friction} 0.01 0.001")

    if DECAL_PATH.exists():
        _attach_plate_decal(plate, root, half_x, half_y, half_thick)

    plate_site = ET.SubElement(plate, "site")
    plate_site.set("name", "plate_center")
    plate_site.set("pos", "0 0 0.01")
    plate_site.set("size", "0.005")


def _attach_plate_decal(
    plate: ET.Element,
    root: ET.Element,
    half_x: float,
    half_y: float,
    half_thick: float,
) -> None:
    """Overlay a thin textured slab on top of the plate (logo + course name)."""
    asset = root.find("asset") or ET.SubElement(root, "asset")
    if asset.find("texture[@name='plate_decal']") is None:
        tex = ET.SubElement(asset, "texture")
        tex.set("name", "plate_decal")
        tex.set("type", "2d")
        tex.set("file", str(DECAL_PATH))
        mat = ET.SubElement(asset, "material")
        mat.set("name", "plate_decal_mat")
        mat.set("texture", "plate_decal")
        mat.set("texuniform", "false")
        mat.set("texrepeat", "1 1")

    decal = ET.SubElement(plate, "geom")
    decal.set("type", "box")
    decal.set("size", f"{half_x} {half_y} 0.0005")
    decal.set("pos", f"0 0 {half_thick + 0.0008}")
    decal.set("mass", "0.001")
    decal.set("material", "plate_decal_mat")
    # Wood-coloured base for transparent regions of the texture.
    decal.set("rgba", "0.86 0.72 0.50 1")
    # Decal must not collide with anything; it is purely visual.
    decal.set("contype", "0")
    decal.set("conaffinity", "0")


def _attach_ball(worldbody: ET.Element, cfg: SceneConfig) -> None:
    """Add a free-joint ball above the world origin (will be reseated at reset)."""
    ball = ET.SubElement(worldbody, "body")
    ball.set("name", "ball")
    ball.set("pos", "0 0 2")

    joint = ET.SubElement(ball, "joint")
    joint.set("name", "ball_free")
    joint.set("type", "free")

    geom = ET.SubElement(ball, "geom")
    geom.set("name", "ball")
    geom.set("type", "sphere")
    geom.set("size", str(cfg.ball_radius))
    geom.set("mass", str(cfg.ball_mass))
    geom.set("rgba", "0.9 0.2 0.2 1")
    geom.set("friction", f"{cfg.ball_friction} 0.01 0.001")


def _ensure_light(worldbody: ET.Element) -> None:
    """Add a default top-down white light if the scene has none."""
    if any(el.tag == "light" for el in worldbody):
        return
    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 3")
    light.set("dir", "0 0 -1")
    light.set("diffuse", "0.8 0.8 0.8")


def _ensure_floor(worldbody: ET.Element) -> None:
    """Add a default plane floor if the scene has none."""
    has_floor = any(el.tag == "geom" and el.get("type") == "plane" for el in worldbody)
    if has_floor:
        return
    floor = ET.SubElement(worldbody, "geom")
    floor.set("type", "plane")
    floor.set("size", "5 5 0.1")
    floor.set("rgba", "0.3 0.3 0.3 1")
