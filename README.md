# rlc2ss

## Overview

rlc2ss converts an electrical netlist into a C++ state-space model. The
generated model supports fixed-step simulation, externally controlled
switches, diodes, and saturating inductors.

Internally, a circuit topology is represented by six intermediate matrices,
$K_1$, $K_2$, $A_1$, $B_1$, $C_1$, and $D_1$. The state-space matrices $A$,
$B$, $C$, and $D$ in

$\frac{dx}{dt} =A x + B u$\
$Y =C x + D u$

are calculated as

$A = K_1^{-1} A_1$\
$B = K_1^{-1} B_1$\
$C = (C_1 + K_2 K_1^{-1} A_1)$\
$D = (D_1 + K_2 K_1^{-1} B_1)$

## Build

Install [uv](https://docs.astral.sh/uv/) and
[just](https://github.com/casey/just), then configure and build:

```sh
just setup build
just build
```

Run the tests with:

```sh
just test build
```

The debugging GUI dependency is expected at `subprojects/DbgGui`. Clone DbgGui
next to this repository and create the link before configuring:

```sh
ln -s ../../DbgGui subprojects/DbgGui
```

Meson reports an error during configuration if the directory is missing.

## Usage

Generate a model which constructs and caches switch topologies on demand at
runtime:

```sh
just rlc2ss schematics/RL3.cir --dynamic
```

This is the recommended mode for switched circuits. Generation is fast and
the generated model only constructs topologies that are actually used. The
first use of a topology performs the state-space construction; subsequent
uses are served from the model's cache.

Without `--dynamic`, rlc2ss solves every valid switch combination during
generation and writes the intermediate matrices to JSON:

```sh
just rlc2ss schematics/RL3.cir
```

Static generation can still be useful when all construction work must happen
ahead of runtime, but its generation time and matrix data grow with the number
of switches.

Both modes generate `<netlist_name>_matrices.hpp` and
`<netlist_name>_matrices.cpp`. Static mode additionally generates
`<netlist_name>_matrices.json`, which is embedded into the application at
build time.

## Supported components

- Resistors (R)
- Inductors (L)
- Capacitors (C)
- Voltage source (V)
- Current source (I)
- Voltage controlled voltage source (E)
- Current controlled current source (F)
- Voltage controlled current source (G)
- Current controlled voltage source (H)
- Switch (S)
- Mutual inductance (K). Mutual inductances have to be given after all other components in the netlist. The syntax is K{name} {inductor 1 name} {inductor 2 name}
