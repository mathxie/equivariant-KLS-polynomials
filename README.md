# Equivariant KL for Thagomizer under $B_n$

This repository bundle is a minimal reproduction package for the paper:

**Equivariant Kazhdan--Lusztig Polynomials of Thagomizer Matroids with a Hyperoctahedral Group Action**  
Matthew H. Y. Xie, Philip B. Zhang, Michael X. Zhong  
arXiv: `2602.10646`  
https://arxiv.org/abs/2602.10646

It computes the equivariant Kazhdan--Lusztig coefficients (and optionally the
inverse coefficients) for the thagomizer matroid $T_n$ with the natural
hyperoctahedral action $B_n$, and prints each coefficient as a decomposition
into irreducibles labeled by bipartitions.

Repository layout (minimal reproducible subset):

- `families/tn.sage`
- `core/bases.sage`
- `core/ekl.sage`
- `core/equivariant_characteristic.sage`
- `core/rep2part.sage`

## Requirements

- SageMath (with GAP available via Sage; the default Sage install is OK).

## Quick start

Run in a fresh Sage process:

```bash
git clone https://github.com/mathxie/equivariant-KLS-polynomials
cd equivariant-KLS-polynomials
sage -c "load('families/tn.sage'); analyse_thagomizer(4, verbose=True)"
```

Optional:

```bash
git clone https://github.com/mathxie/equivariant-KLS-polynomials
cd equivariant-KLS-polynomials
sage -c "load('families/tn.sage'); analyse_thagomizer(4, verbose=True, compute_inverse=True)"
```

Output format (each coefficient):

- `t^i dim=... :: [ (lambda) , (mu) ]:mult, ...`

For inverse coefficients:

- `Q t^i dim=... :: [ (lambda) , (mu) ]:mult, ...`

## Main results

Irreducible representations of $B_n$ are labeled by bipartitions $(\lambda,\mu)$
of $n$, written as $V_{\lambda,\mu}$.

For all $n \ge 0$, in $\mathrm{grVRep}(B_n)$:

```math
P_{T_n}^{B_n}(t)
=
V_{(n),\varnothing}
\;+\;
\sum_{k=1}^{\left\lfloor \frac{n}{2}\right\rfloor}
\Bigl(\sum_{i=2k}^{n} V_{(n-i),(i-2k+2,2^{k-1})}\Bigr)t^{k},
```

with the convention $(0)=\varnothing$.

For all $n \ge 0$, in $\mathrm{grVRep}(B_n)$:

```math
Q_{T_n}^{B_n}(t)
=
\sum_{k=0}^{\left\lfloor\frac{n}{2}\right\rfloor}
\Bigl(\sum_{i=2k}^{n} V_{(1^{n-i}),(2^k,1^{i-2k})}\Bigr)t^k.
```

## Paper mapping

- `families/tn.sage`: construction of $T_n$ and the $B_n$-action workflow.
- `core/ekl.sage`: equivariant KL and inverse KL recursion.
- `core/rep2part.sage`: bipartition labeling for $B_n$ irreducibles.

## License

This repository is released under the GNU General Public License,
version 2 or (at your option) any later version (`GPL-2.0-or-later`).
See `LICENSE`.
