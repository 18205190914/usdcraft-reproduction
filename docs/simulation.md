# Simulate a run

Use simulation to check how a generated object moves under gravity. Simulation requires the
optional MuJoCo dependency.

## Install the simulator

Install the simulation dependency group:

```shell
uv sync --group sim
```

## Run a drop test

The default scenario releases the object above a floor. It then simulates three seconds:

```shell
uv run articraft simulate runs/<run-id>
```

The command reports these values:

| Result | Meaning |
| --- | --- |
| `lowest body` | The initial and final height of the lowest body origin. |
| `contacts at rest` | The number of contact points at the end of the test. |
| `deepest penetration` | The largest overlap on impact and at rest. |
| `largest part separation change` | The largest change in distance between two bodies. |
| `residual velocity` | The highest body speed at the end of the test. |
| `verdict` | Whether the object stayed together and came to rest. |

A body origin can be above or below its geometry. Use contact and penetration values to find
floor failures. Do not use the body origin height alone.

## Test sliding friction

Use the `tilt` scenario to raise the floor until the object slides:

```shell
uv run articraft simulate runs/<run-id> --scenario tilt --seconds 8
```

The report includes the slip angle and a comparison with the authored friction value.

## Release the joints

Use the `release` scenario to start each joint at the middle of its range:

```shell
uv run articraft simulate runs/<run-id> --scenario release
```

The report includes the highest joint speed during the test.

## Play the recorded motion

Simulation writes a trajectory record under `result/simulation/`. Open the run to play it:

```shell
uv run articraft view runs/<run-id>
```

Select **Play simulation** in the viewer.

## Know the limits

A passing run checks geometry, mass, joint behavior, and sliding friction. It does not check
restitution or separate static friction.

MuJoCo uses one sliding friction value and has no restitution parameter. Articraft still
exports those authored values for other engines.
