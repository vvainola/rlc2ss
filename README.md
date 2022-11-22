# RLC_to_state_space

## Overview
Formulate RLC circuit from netlist into state-space form. The output matrices are given as 6 intermediate matrices $K_1$, $K_2$, $A_1$, $B_1$, $C_1$ and $D_1$
to avoid inverting the matrices in symbolic form as this can be too difficult for sympy to handle.

The state-space form matrices $A$, $B$, $C$ and $D$ in

$\frac{dx}{dt} =A x + B u$\
$Y =C x + D u$

can be calculated from the intermediate matrices as

$A = K_1^{-1} A_1$\
$B = K_1^{-1} B_1$\
$C = (C_1 + K_2 K_1^{-1} A_1)$\
$D = (D_1 + K_2 K_1^{-1} B_1)$

## Usage
`
python scripts/rlc2ss.py schematics/RL3.cir
`

The script will make a C++ header {netlist_name}_matrices.h that contains the matrices.

## Supported components
- Resistors (R)
- Inductors (L)
- Capacitor (C)
- Voltage source (V)
- Current source (I)
- Voltage controlled voltage source (E)
- Current controlled current source (F)
- Voltage controlled current source (G)
- Current controlled voltage source (H)
- Switch (S). All 2^n combinations of switches are created.

- Mutual inductance (K). Mutual inductances have to be given after all other components in the netlist. The syntax is K{name} {inductor 1 name} {inductor 2 name}
