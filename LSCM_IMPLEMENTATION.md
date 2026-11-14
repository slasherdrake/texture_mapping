# Least Squares Conformal Maps

`src/param/lscm.py` implements the conformal energy from
[Lévy et al., SIGGRAPH 2002](https://brunolevy.github.io/papers/LSCM_SIGGRAPH_2002.pdf).
Triangle coordinate frames, gradients, weights, constraints, and sparse matrix
assembly are implemented here. SciPy supplies the generic LSQR solver, not a
parameterization algorithm.

## Triangle equations

Place each oriented triangle in local coordinates
`p0 = (0, 0)`, `p1 = (L, 0)`, `p2 = (x, y)`, where `y > 0`.
The gradients of its three linear basis functions are:

```text
gx = [-1/L,           1/L,        0]
gy = [(x/L - 1)/y,   -x/(L*y),   1/y]
```

The Cauchy-Riemann equations require `u_x = v_y` and `u_y = -v_x`.
For each triangle of area `A`, add these two residual rows:

```text
sqrt(A) * [gx, -gy]
sqrt(A) * [gy,  gx]
```

With unknowns ordered as `[u0, ..., uN-1, v0, ..., vN-1]`, the resulting sparse
matrix `M` has shape `(2 * number_of_faces, 2 * number_of_vertices)`.
Minimizing `||M @ UV||²` minimizes the integrated squared conformality error.
Area weighting matters: triangles of different sizes must contribute according
to surface area, not just triangle count.

## Pins and solve

Two boundary vertices fix translation, rotation, and scale. By default, two
Euclidean farthest-point sweeps on the boundary choose separated pins, mapped
to `(0, 0)` and `(1, 0)`. This is a linear-time heuristic, not an exact diameter
or geodesic computation. Explicit pins and UV targets are also supported.

Eliminate the four fixed scalar variables and solve
`M_free @ UV_free ≈ -M_fixed @ UV_fixed`. LSQR solves this rectangular system
without explicitly forming normal equations. The implementation checks solver
termination and restores the pin coordinates exactly. All remaining vertices,
including boundary vertices, are free. Output is not independently rescaled along
the two axes, since that would introduce conformal distortion and change the pins.

## Mesh assumptions

Validation checks finite geometry, integer face indices, used vertices, duplicate
faces, edge incidence, consistent winding, connectivity, a single boundary loop,
Euler characteristic one, and nondegenerate triangle areas. A malformed input
raises `ValueError` before solving.

The generated disk and hemisphere previously had ring triangles wound opposite
to their center/pole fans. Their ring winding is corrected so shared edges have
opposite directions in adjacent triangles. This is required for a coherent LSCM
orientation and also corrects their face normals.

Closed surfaces require cutting first. Improving the existing experimental cut
routine and making charts for multiple-boundary or disconnected OBJ meshes are
separate work. The clean viewer offers LSCM only for meshes that pass validation.
The energy does not explicitly constrain triangle signs, global overlap, or area
preservation; inspect distortion metrics rather than assuming every result is
injective.

## Validation

```bash
python -m pytest tests -q
```

Tests cover a planar patch's exact similarity map, custom constraints, rigid
rotation and scale in 3D, a single triangle, automatic pins, near-unit planar
angle distortion, curved-surface energy compared to Tutte under identical pins,
invalid inputs, and both viewers' registration and method selection without
opening a graphical window.
