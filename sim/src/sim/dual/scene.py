"""Compose a side-by-side dual UR5e MuJoCo scene.

Calls :func:`sim.scene.build_scene` to produce the canonical single arm
+ plate + ball model, dumps the resulting XML via ``mj_saveLastXML``,
then deep-copies the arm and ball subtrees with a ``_b`` suffix and
shifts them along the world ``x`` axis. ``sim.scene`` itself is never
modified; the entire two-arm logic is local to this module.

Element naming after composition::

    arm A  unchanged   ("base", "shoulder_pan_joint", "plate_center", "ball", ...)
    arm B  ``_b`` suffix ("base_b", "shoulder_pan_joint_b", "plate_center_b",
                          "ball_b", ...)

The runner looks up arm-specific bodies, joints, actuators, and sites
by name plus suffix so the implementation never hard-codes numeric IDs.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass

import mujoco

from sim.scene import SceneConfig, build_scene

#: Suffix appended to every name in the duplicated subtree (and to any
#: reference attribute pointing into that subtree). Kept as a single source
#: of truth so the runner can rebuild the same lookups by name.
ARM_B_SUFFIX: str = "_b"

#: XML attributes that hold a reference to another named element. The
#: dual builder rewrites them when the referenced element was renamed.
_REF_ATTRS = ("name", "joint", "site", "body", "body1", "body2", "objname")


@dataclass(frozen=True)
class DualSceneConfig:
    """Wrapper around the single-arm config plus the inter-arm spacing."""

    base: SceneConfig
    base_offset_x: float = 0.6  # half of the inter-arm distance, in metres


def build_dual_scene(cfg: DualSceneConfig) -> mujoco.MjModel:
    """Compose two UR5e arms in one MjModel and return it.

    Both arms share the floor, the lighting, and the asset catalogue; only
    the kinematic tree, the contact excludes, and the actuators are
    duplicated. The shared ``/tmp/ur5e_*_<pid>.xml`` filenames keep this
    safe for parallel processes.
    """
    single = build_scene(cfg.base)
    pid_str = str(os.getpid())
    single_xml = f"/tmp/ur5e_plate_ball_{pid_str}.xml"
    mujoco.mj_saveLastXML(single_xml, single)

    tree = ET.parse(single_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Single composed scene has no <worldbody>.")

    arm_a = _find_body_by_name(worldbody, "base")
    ball_a = _find_body_by_name(worldbody, "ball")
    if arm_a is None or ball_a is None:
        raise RuntimeError("Single scene missing 'base' and/or 'ball' body.")

    # Position arm A at -offset; ball A starts above arm A.
    arm_a.set("pos", f"{-cfg.base_offset_x} 0 0")
    ball_a.set("pos", f"{-cfg.base_offset_x} 0 2")

    # Collect every name defined inside the duplicated subtrees so that
    # reference attributes (joint=, body1=, ...) are rewritten consistently.
    duplicated_names: set[str] = set()
    _collect_names(arm_a, duplicated_names)
    _collect_names(ball_a, duplicated_names)
    actuator_block = root.find("actuator")
    contact_block = root.find("contact")
    if actuator_block is not None:
        for entry in actuator_block:
            if "name" in entry.attrib:
                duplicated_names.add(entry.attrib["name"])

    # Duplicate the arm tree.
    arm_b = deepcopy(arm_a)
    _rename_subtree(arm_b, ARM_B_SUFFIX, duplicated_names)
    arm_b.set("pos", f"{cfg.base_offset_x} 0 0")
    worldbody.append(arm_b)

    # Duplicate the ball.
    ball_b = deepcopy(ball_a)
    _rename_subtree(ball_b, ARM_B_SUFFIX, duplicated_names)
    ball_b.set("pos", f"{cfg.base_offset_x} 0 2")
    worldbody.append(ball_b)

    # Duplicate the actuators (one per arm-B joint).
    if actuator_block is not None:
        for entry in list(actuator_block):
            new_entry = deepcopy(entry)
            _rename_subtree(new_entry, ARM_B_SUFFIX, duplicated_names)
            actuator_block.append(new_entry)

    # Duplicate the contact <exclude> entries so arm B has the same self-
    # collision filtering arm A enjoys.
    if contact_block is not None:
        for excl in list(contact_block):
            new_excl = deepcopy(excl)
            _rename_subtree(new_excl, ARM_B_SUFFIX, duplicated_names)
            contact_block.append(new_excl)

    composed_xml = f"/tmp/ur5e_dual_{pid_str}.xml"
    tree.write(composed_xml)
    return mujoco.MjModel.from_xml_path(composed_xml)


# ============================================================================
# XML helpers (private; kept simple: no recursion into asset definitions)
# ============================================================================


def _find_body_by_name(elem: ET.Element, name: str) -> ET.Element | None:
    """Depth-first search for a ``<body name="...">`` matching ``name``."""
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for child in elem:
        result = _find_body_by_name(child, name)
        if result is not None:
            return result
    return None


def _collect_names(elem: ET.Element, names: set[str]) -> None:
    """Recursively gather all ``name`` attributes inside ``elem``."""
    if "name" in elem.attrib:
        names.add(elem.attrib["name"])
    for child in elem:
        _collect_names(child, names)


def _rename_subtree(
    elem: ET.Element,
    suffix: str,
    original_names: set[str],
) -> None:
    """Append ``suffix`` to ``name`` attributes and to references targeting them.

    Reference attributes are renamed only when their value matches an entry
    in ``original_names`` so we don't accidentally rewrite class/material/
    mesh references that point into the shared <asset> catalogue.
    """
    for attr in _REF_ATTRS:
        if attr in elem.attrib:
            old = elem.attrib[attr]
            if attr == "name" or old in original_names:
                elem.set(attr, old + suffix)
    for child in elem:
        _rename_subtree(child, suffix, original_names)
