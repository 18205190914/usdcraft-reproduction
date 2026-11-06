# Articraft SDK

Use the Articraft SDK to build articulated 3D objects in Python. You can combine build123d
solids with procedural triangle meshes.

Import the public object API from `articraft.sdk`. Import mesh operations from
`articraft.sdk.mesh`.

## Use the correct units

Use meters for all lengths and coordinates. Use radians for joint rotations and mesh
rotations. Build123d rotations use degrees.

The SDK uses right handed XYZ coordinates with Z as the up axis.

## Create an object

A `RigidBodyAssembly` contains the rigid bodies in one object. Put shapes in the same body
when they always move together.

```python
from build123d import Box

from articraft.sdk import RigidBodyAssembly, TestContext
from articraft.sdk.export import export_assembly

model = RigidBodyAssembly("box")
body = model.rigid_body("body")
body.add(Box(0.1, 0.1, 0.1), name="shell")

model.validate()
report = TestContext(model).report()
assert report.passed

result = export_assembly(model, "output")
print(result.usdz)
```

Give each shape a unique name within its body. You can also give a shape an RGB or RGBA color.

## Select a geometry type

Use build123d when you need exact solids, cuts, fillets, faces, or edges. Add a completed
build123d shape directly to a rigid body.

Use `MeshGeometry` when you need direct vertex changes or a procedural surface. The mesh API
also provides booleans, sweeps, lofts, shells, welds, and smoothing.

```python
from articraft.sdk import RoundedBoxGeometry

housing = RoundedBoxGeometry((0.12, 0.075, 0.028), radius=0.006)
body.add(housing, name="housing", color=(0.25, 0.30, 0.36))
```

Both geometry types use body local coordinates. Convert a build123d shape only when a mesh
operation needs it.

## Add motion

A joint connects two rigid bodies. Its degrees of freedom specify which axes can move.

An articulation defines the joint tree that the solver uses:

```python
from build123d import Box

from articraft.sdk import JointAxis, JointDOF

lid = model.rigid_body("lid")
lid.add(Box(0.1, 0.1, 0.01), name="panel")

model.joint(
    "body_to_lid",
    body.at((0.0, 0.05, 0.05)),
    lid.at(),
    dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.8)),),
)
model.articulation("main", root="body", joints=["body_to_lid"])
```

The two joint frames coincide in the reference state. Joint limits must include zero.

Use `body.at(...)` to bind a point or build123d feature to its body. For a closed loop, author
every physical joint. Omit one closing joint from the articulation tree.

## Check the object

Call `model.validate()` before export. It checks names, geometry, joints, articulation trees,
and the reference state.

Use `TestContext` to check distances, overlaps, support, poses, and motion. The compiler also
checks for isolated bodies and disconnected geometry.

The compiler reports scale problems, unwanted overlap, and joints that separate during
motion.

## Export the object

Import `export_assembly(...)` from `articraft.sdk.export`. This separate import prevents a
normal SDK import from loading OpenUSD.

The exporter writes a USDZ file and a schema version 2 manifest. Each rigid body becomes one
USD rigid body.

## Open the reference

Start with the [SDK quickstart](../src/articraft/sdk/docs/common/00_quickstart.md).

### Object structure and motion

- [Shared units and types](../src/articraft/sdk/docs/common/20_core_types.md)
- [Assemblies and rigid bodies](../src/articraft/sdk/docs/common/30_assembly.md)
- [Joints and articulations](../src/articraft/sdk/docs/common/35_joints.md)
- [Materials and mass](../src/articraft/sdk/docs/common/37_materials.md)
- [Simulation settings](../src/articraft/sdk/docs/common/38_simulation_settings.md)

### Checks and output

- [Errors](../src/articraft/sdk/docs/common/10_errors.md)
- [Testing](../src/articraft/sdk/docs/common/40_testing.md)
- [Visual evidence](../src/articraft/sdk/docs/common/45_visual_evidence.md)
- [USDZ export](../src/articraft/sdk/docs/common/50_usdz_export.md)

### Mesh authoring

- [Mesh geometry and solid builders](../src/articraft/sdk/docs/mesh/00_mesh_geometry.md)
- [Profiles and curve sampling](../src/articraft/sdk/docs/mesh/10_profiles.md)
- [Wires, pipes, and sweeps](../src/articraft/sdk/docs/mesh/20_wires_and_sweeps.md)
- [Section lofts](../src/articraft/sdk/docs/mesh/30_section_lofts.md)
- [Booleans and shells](../src/articraft/sdk/docs/mesh/40_booleans_and_shells.md)
- [Refinement and smoothing](../src/articraft/sdk/docs/mesh/50_refinement_and_smoothing.md)

You can also read the [complete examples](../src/articraft/sdk/docs/examples) and the
[vendored build123d reference](../src/articraft/sdk/docs/build123d).
