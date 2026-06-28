## About

In this repository one can find every file that is related with the creation (or modification) of Shallow Water Moment Models with Rainfall and Infiltration dynamics[^1]. The repository has the following structure:

```text
.
├── kinetic_sw
├── moment_sw
├── processing
├── symbolic_math
└── utils
```

In addition, it is also standard practice that executable files are never uploaded to GitHub repositoties. As such, files like figures, graphs, images, executables, binaries, etc..., are not uploaded here. However all of these are stored locally, and access to them can be provided upon request.

## `kinetic_sw/`
The now deprecated first attempt to build a kinetic solver on the subject of Shallow Water Moment Equations, after being inspired by Ersoy et al. This code works, but it lacking compared to the solver that is found on `moment_sw/`.

## `moment_sw/`
This folder contains the Python solver of the Shallow Water Moment Equations for Rainfall-Runoff, as well as other implementations. My contribution to this is the create of the `recharge/` module, as well as the modification of the original files to wrap successfully with the new rainfall-runoff specific physics.

## `processing/`
All post-processing scripts, templates and solver configurations are stored here.

## `symbolic_math/`
Introducing `SYMBO.py`, a very efficient Python script that performs symbolic integration to derive explicit values of the coefficients that define the Shallow Water Moment Equations (+Rainfall-Runoff).

## `utils/`
Helper Python and Bash scripts for navigating and backing important data.

[^1]: Some parts of the project are on purpose left incomplete to avoid personal data leakage to Microsoft as much as possible.