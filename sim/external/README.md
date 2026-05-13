# sim/external

Runtime-cloned third-party assets. Every subdirectory is gitignored.

| Path                | Upstream                                            | Consumer    |
| ------------------- | --------------------------------------------------- | ----------- |
| `mujoco_menagerie/` | https://github.com/google-deepmind/mujoco_menagerie | `sim.scene` |

## Provisioning

`sim.scene.build_scene` clones any missing entry on first invocation and
reuses the local copy thereafter. Pinning is not enforced; switch to a
submodule against a release tag if the project requires reproducibility
across an upstream change.
