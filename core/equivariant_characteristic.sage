"""
equivariant_characteristic.sage
-------------------------------

Purpose / public entry points
-----------------------------

This file implements the equivariant Möbius invariant and the equivariant characteristic
polynomial in three independent ways (useful for cross-checks):

- ``equivariant_characteristic_polynomial(X, W=None)``: dispatcher
  - Matroid input -> OS-global (ground-set action)
  - Poset input   -> incidence-algebra / Möbius (poset action)
- ``equivariant_characteristic_poset(P, W=None)``: incidence-algebra method for a bounded poset
- ``equivariant_characteristic_mobius(M, W_ground=None)``: inflate the poset result to a ground-set action
- ``equivariant_characteristic_os_local(M, W=None)``
- ``equivariant_characteristic_os_global(M, W=None)``
- ``equivariant_mobius(P, W=None)``, ``equivariant_mu_interval_top(P, W=None)``

Notes
-----
- The three implementations can disagree on matroids with loops; for cross-checks prefer loopless
  matroids (or delete loops before comparing with OS/Sage characteristic polynomials).

Implementation overview
-----------------------

1) **Möbius / incidence algebra** (Proudfoot): cached recursion on canonicalised lower intervals.
2) **Local Orlik–Solomon** (Brieskorn): localisation at flats.
3) **Global Orlik–Solomon**: graded characters of the global OS algebra.

Dependencies:
  - ``core/bases.sage`` (group/poset helpers)

SageMath examples
-----------------

EXAMPLES::

    sage: load('core/equivariant_characteristic.sage')
    sage: M = matroids.Uniform(3, 5)
    sage: H = equivariant_characteristic_polynomial(M)  # OS-global by default
    sage: G = M.automorphism_group()
    sage: [cf(G.identity()) for cf in H]
    [-6, 10, -5, 1]
    sage: H2 = equivariant_characteristic_mobius(M)     # incidence/Möbius (inflated)
    sage: [cf(G.identity()) for cf in H2] == [cf(G.identity()) for cf in H]
    True
"""

import os
import inspect
import sys


def _this_sage_file_path():
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0.endswith(".sage") and os.path.exists(argv0):
        return os.path.abspath(argv0)
    for fr in inspect.stack():
        for key in ("fpath", "filename"):
            val = fr.frame.f_locals.get(key)
            if val is None:
                continue
            try:
                val = os.fspath(val)
            except TypeError:
                continue
            if isinstance(val, str) and val.endswith(".sage") and os.path.exists(val):
                return os.path.abspath(val)
    return None


_THIS_SAGE_FILE = _this_sage_file_path()
_CORE_DIR = os.path.dirname(_THIS_SAGE_FILE) if _THIS_SAGE_FILE else None
if _CORE_DIR:
    load(os.path.join(_CORE_DIR, "bases.sage"))
else:
    load("core/bases.sage")

# =============================================================================
# PART 1: Möbius / Incidence Algebra (Proudfoot)
# =============================================================================

def _trivial_cf(group):
    """
    Trivial (constant 1) class function on ``group``.

    Parameters
    ----------
    group : finite group

    Returns
    -------
    ClassFunction
    """

    return ClassFunction(group, [1] * len(group.conjugacy_classes()))

_MU_CACHE = {}

def _interval_key(poset_can, group_can):
    """
    Cache key for equivariant μ on a canonical interval (P_can, W_can).

    The key includes:
    - the canonical poset (via its graph6 string),
    - the canonical generators (as permutations of the poset domain indices),
    - the canonical conjugacy-class representatives (also as permutations of indices).

    Including conjugacy-class reps fixes a subtle Sage/GAP pitfall: two equal
    permutation groups can still enumerate conjugacy classes in different
    orders across processes, which would otherwise corrupt the cache.
    """

    g6 = poset_can.hasse_diagram().to_undirected().graph6_string()
    domain = tuple(poset_can)
    idx = {x: i for i, x in enumerate(domain)}
    gens = tuple(tuple(idx[g(x)] for x in domain) for g in group_can.gens())
    reps = tuple(
        tuple(idx[rep(x)] for x in domain)
        for rep in group_can.conjugacy_classes_representatives()
    )
    return g6, gens, reps

def _mu_interval_canonical(poset_can, group_can):
    """
    Compute equivariant μ_{[0,top]} on a canonical (poset, group) pair.

    Parameters
    ----------
    poset_can : Poset
        Canonicalised bounded poset representing the interval.
    group_can : PermutationGroup
        Canonicalised group acting on ``poset_can``.

    Returns
    -------
    ClassFunction
        The equivariant Möbius class function on ``group_can``.
    """

    key = _interval_key(poset_can, group_can)
    cached = _MU_CACHE.get(key)
    if cached is not None:
        return ClassFunction(group_can, cached)

    if poset_can.rank() == 0:
        mu = _trivial_cf(group_can)
        _MU_CACHE[key] = mu.values()
        return mu

    bottom = poset_can.bottom()
    top = poset_can.top()
    total = ClassFunction(group_can, [0] * len(group_can.conjugacy_classes()))

    for y in poset_can:
        if y == top:
            continue
        if not poset_can.le(bottom, y):
            continue

        elements = [x for x in poset_can if poset_can.le(x, y)]
        sub_poset = poset_can.subposet(elements)
        stabiliser = group_can.stabilizer(y)

        mu_sub = _mu_interval_top(sub_poset, stabiliser)
        if stabiliser.order() == group_can.order():
            reps = group_can.conjugacy_classes_representatives()
            induced = ClassFunction(group_can, [mu_sub(g) for g in reps])
        else:
            induced = mu_sub.induct(group_can)

        weight = QQ(stabiliser.order()) / QQ(group_can.order())
        total += weight * induced

    mu = (-1) * total
    _MU_CACHE[key] = mu.values()
    return mu

def _mu_interval_top(poset, group):
    """
    Compute μ_{[0,top]} for an arbitrary (poset, group) action using canonical caching.

    This performs:
      shrink_group_to_poset -> canonicalize_action -> cached recursion -> inflate back.

    Returns
    -------
    ClassFunction
        Class function on the original input group.
    """

    group_small, pi = shrink_group_to_poset(group, poset)
    poset_can, group_can, relabel, phi = canonicalize_action(poset, group_small)

    mu_can = _mu_interval_canonical(poset_can, group_can)
    mu_small = inflate_classfunction(mu_can, phi)
    mu_original = inflate_classfunction(mu_small, pi)
    return mu_original


def equivariant_mu_interval_top(P, W=None):
    """
    Equivariant Möbius invariant μ_{[0,1]}^W for the top interval of P.

    This is a public wrapper around the cached interval recursion used
    internally by the characteristic / inverse-Z pipelines.

    Parameters
    ----------
    P : Poset or matroid
        If P is a matroid, we use its lattice of flats.
    W : PermutationGroup or None
        - If P is a poset: W acts on the poset elements. If None, use the
          automorphism group of the Hasse diagram.
        - If P is a matroid: W is interpreted as a ground-set action (like
          `ekl(M, W)`); we compute the induced action on flats internally and
          inflate μ back to the ground action. If None, use `Aut(M)` on the
          ground set.

        Backwards compatibility: if P is a matroid and the provided W does not
        act on the ground set, we treat W as a flats action on `L(M)` (poset
        elements are flats).
    """

    if hasattr(P, "lattice_of_flats"):
        M = P
        L = M.lattice_of_flats()

        if W is None:
            W_ground = M.automorphism_group()
            W_flats = induced_action_on_flats_group(M, W_ground)
            pi = induced_action_on_flats(W_ground, W_flats)
            mu_flats = _mu_interval_top(L, W_flats)
            return inflate_classfunction(mu_flats, pi, W_ground, W_flats)

        ground = list(M.groundset())
        try:
            ground_set = set(ground)
            dom_in_ground = set(W.domain()).issubset(ground_set)
        except TypeError:
            dom_in_ground = all(x in ground for x in W.domain())

        if not dom_in_ground:
            return _mu_interval_top(L, W)

        # Extend a partial ground-set action by fixing missing points, to make
        # `induced_action_on_flats_group` safe.
        target = ground
        try:
            target = sorted(target)
        except Exception:
            target = sorted(target, key=lambda e: repr(e))

        domain = list(W.domain())
        domain_set = set(domain)
        target_set = set(target)
        if not set(target).issubset(domain_set):
            images = []
            for sigma in W.gens():
                image = []
                for e in target:
                    if e in domain_set:
                        img = sigma(e)
                    else:
                        img = e
                    if img not in target_set:
                        raise ValueError("W moves an element outside the ground set.")
                    image.append(img)
                images.append(image)
            W_act = PermutationGroup(gens=images, domain=target)
            images_as_elements = [W_act(g_img) for g_img in images]
            inj = W.hom(images_as_elements, W_act)
        else:
            W_act = W
            inj = None

        W_flats = induced_action_on_flats_group(M, W_act)
        pi = induced_action_on_flats(W_act, W_flats)
        mu_flats = _mu_interval_top(L, W_flats)
        mu_act = inflate_classfunction(mu_flats, pi, W_act, W_flats)
        if inj is None:
            return mu_act
        return inflate_classfunction(mu_act, inj, W, W_act)

    if W is None:
        W = P.hasse_diagram().automorphism_group()
    return _mu_interval_top(P, W)

def equivariant_mobius(P, W=None):
    """
    Equivariant Möbius data for a bounded poset (or a lattice of flats).

    This is the shared entry point used by the incidence-algebra characteristic
    polynomial builder.  It returns the canonicalised action together with the
    Möbius class functions on stabilisers of canonical representatives.

    INPUT:

    - ``P`` -- a bounded poset (or a matroid, in which case we use ``P.lattice_of_flats()``).
    - ``W`` -- (default: ``None``) a permutation group acting on the poset elements.
      If ``None``, use automorphisms of the Hasse diagram.

    OUTPUT:

    A 6-tuple ``(mu_data, P_can, W_can, phi, pi, relabel)`` where:

    - ``mu_data`` is a dict mapping each ``F`` in ``P_can`` to ``(W_F, mu_F)``, where
      ``W_F`` is the stabiliser of ``F`` in ``W_can`` and ``mu_F`` is the equivariant
      Möbius class function of the lower interval ``[0,F]`` on ``W_F``;
    - ``P_can, W_can, relabel, phi`` come from ``canonicalize_action`` on the shrunk group;
    - ``pi`` is the shrink map from the original group to the shrunk group.

    EXAMPLES::

        sage: load('core/equivariant_characteristic.sage')
        sage: P = Poset(([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)]))  # Boolean lattice B_2
        sage: mu_data, P_can, W_can, phi, pi, relabel = equivariant_mobius(P)
        sage: H = equivariant_characteristic_from_mobius_data(mu_data, P_can, W_can, phi, pi)
        sage: W = P.hasse_diagram().automorphism_group()
        sage: [cf(W.identity()) for cf in H]
        [1, -2, 1]
    """
    if hasattr(P, "lattice_of_flats"):
        P = P.lattice_of_flats()
    if W is None:
        W = P.hasse_diagram().automorphism_group()

    W_small, pi = shrink_group_to_poset(W, P)
    P_can, W_can, relabel, phi = canonicalize_action(P, W_small)

    mu_data = {}
    for F in sorted(P_can, key=lambda x: P_can.rank(x)):
        stabiliser = W_can.stabilizer(F)
        elements = [x for x in P_can if P_can.le(x, F)]
        sub_poset = P_can.subposet(elements)
        mu_F = _mu_interval_top(sub_poset, stabiliser)
        mu_data[F] = (stabiliser, mu_F)

    return mu_data, P_can, W_can, phi, pi, relabel

def equivariant_characteristic_from_mobius_data(mu_data, P_can, W_can, phi, pi):
    """
    Assemble equivariant characteristic coefficients from ``equivariant_mobius`` output.

    This helper keeps the incidence-algebra (μ-based) characteristic computation
    factored out so that callers who already computed μ-data can avoid redoing
    interval recursion.

    Parameters
    ----------
    mu_data, P_can, W_can, phi, pi
        As returned by ``equivariant_mobius(P, W)``.

    Returns
    -------
    list[ClassFunction]
        Coefficients [H_0, ..., H_r] as class functions on the original group W.
    """
    rank = P_can.rank()

    coeffs_can = [
        ClassFunction(W_can, [0] * len(W_can.conjugacy_classes()))
        for _ in range(rank + 1)
    ]

    for F, (stabiliser, mu_F) in mu_data.items():
        deg = rank - P_can.rank(F)
        if stabiliser.order() == W_can.order():
            reps = W_can.conjugacy_classes_representatives()
            induced = ClassFunction(W_can, [mu_F(g) for g in reps])
        else:
            induced = mu_F.induct(W_can)
        weight = QQ(stabiliser.order()) / QQ(W_can.order())
        coeffs_can[deg] += weight * induced

    # Constant term H_0 is the equivariant Möbius invariant μ(top).
    top = P_can.top()
    stabiliser_top, mu_top = mu_data[top]
    if stabiliser_top.order() == W_can.order():
        reps = W_can.conjugacy_classes_representatives()
        H0_can = ClassFunction(W_can, [mu_top(g) for g in reps])
    else:
        H0_can = mu_top.induct(W_can)
    coeffs_can[0] = H0_can

    result = []
    for cf in coeffs_can:
        cf_small = inflate_classfunction(cf, phi)
        cf_big = inflate_classfunction(cf_small, pi)
        result.append(cf_big)
    return result

def equivariant_characteristic_poset(P, W=None):
    """
    Equivariant characteristic polynomial via equivariant Möbius data (Lattice Action).
    This implementation uses Proudfoot's incidence algebra formalism and is suitable
    for any bounded graded poset.

    INPUT:

    - ``P`` -- a bounded poset.
    - ``W`` -- (default: ``None``) a permutation group acting on the poset elements.

    OUTPUT:

    A list of class functions ``[H_0, ..., H_r]`` on ``W`` (or the default automorphism
    group when ``W`` is ``None``), representing the equivariant characteristic polynomial.

    EXAMPLES::

        sage: load('core/equivariant_characteristic.sage')
        sage: P = Poset(([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)]))
        sage: H = equivariant_characteristic_poset(P)
        sage: W = P.hasse_diagram().automorphism_group()
        sage: [cf(W.identity()) for cf in H]
        [1, -2, 1]
    """
    mu_data, P_can, W_can, phi, pi, _ = equivariant_mobius(P, W)
    return equivariant_characteristic_from_mobius_data(mu_data, P_can, W_can, phi, pi)

def equivariant_characteristic_polynomial(X, W=None):
    """
    Compute the equivariant characteristic polynomial of X with respect to W.

    Dispatcher:
    - If X is a Matroid (has 'groundset' and 'lattice_of_flats'), uses the 
      Global Orlik-Solomon (Wedge) method by default, as it is generally the fastest
      and operates on the ground set action.
    - Otherwise (e.g. X is a Poset), uses the Möbius / Incidence Algebra method
      (Proudfoot's formalism) operating on the lattice action.
    
    Parameters
    ----------
    X : Matroid or Poset
    W : PermutationGroup, optional
        - If X is a Matroid, W should act on the ground set.
        - If X is a Poset, W should act on the poset elements.
        - Defaults to the full automorphism group if None.
    
    Returns
    -------
    list[ClassFunction]
        The equivariant characteristic polynomial coefficients [H_0, ..., H_r].

    Notes
    -----
    - For a matroid with loops, Sage's (non-equivariant) characteristic
      polynomial is 0. The OS-based implementations here match that convention.
    - If you pass the lattice of flats ``L(M)`` as a Poset, the Möbius/incidence
      method computes the characteristic data of that bounded poset (whose bottom
      element is ``cl(∅)`` = the set of loops), which need not agree with the
      matroid characteristic polynomial.

    EXAMPLES::

        sage: load('core/equivariant_characteristic.sage')
        sage: M = matroids.Uniform(3, 5)
        sage: H = equivariant_characteristic_polynomial(M)  # OS-global by default
        sage: G = M.automorphism_group()
        sage: [cf(G.identity()) for cf in H]
        [-6, 10, -5, 1]
    """
    if hasattr(X, "groundset") and hasattr(X, "lattice_of_flats"):
        return equivariant_characteristic_os_global(X, W)
    else:
        return equivariant_characteristic_poset(X, W)

def induced_action_on_flats(G, W):
    """
    Compute the homomorphism pi: G -> W describing the induced action on flats.
    """
    flats = list(W.domain())
    images = [
        [frozenset(sigma(x) for x in F) for F in flats]
        for sigma in G.gens()
    ]
    return G.hom(images, W)

def equivariant_characteristic_mobius(M, W_ground=None):
    """
    Compute equivariant characteristic coefficients via Möbius, inflated to the
    ground-set group W_ground. (Method 1)

    Note: for loop matroids, this computes the incidence/Möbius characteristic
    of the lattice of flats (bottom = cl(∅)), which may differ from the matroid
    characteristic polynomial (OS-global/local return 0 when a loop is present).

    EXAMPLES::

        sage: load('core/equivariant_characteristic.sage')
        sage: M = matroids.Uniform(3, 5)
        sage: H1 = equivariant_characteristic_mobius(M)
        sage: H2 = equivariant_characteristic_os_global(M)
        sage: G = M.automorphism_group()
        sage: [cf(G.identity()) for cf in H1] == [cf(G.identity()) for cf in H2]
        True
    """
    if W_ground is None:
        W_ground = M.automorphism_group()
        
    P = M.lattice_of_flats()

    W_flats = induced_action_on_flats_group(M, W_ground)
    pi = induced_action_on_flats(W_ground, W_flats)

    coeffs_flats = equivariant_characteristic_poset(P, W_flats)
    
    coeffs_ground = [
        inflate_classfunction(cf, pi) for cf in coeffs_flats
    ]
    return coeffs_ground


# =============================================================================
# PART 2: Local Orlik–Solomon (Brieskorn)
# =============================================================================

def _os_local_localisation_at_flat(M, F):
    """
    Localise the matroid at a flat F for the OS-local method.

    Implementation detail: for a flat ``F`` (subset of the ground set),
    we delete the complement so the resulting matroid has ground set ``F``.
    """

    ground = list(M.groundset())
    delete_ground = [e for e in ground if e not in F]
    if delete_ground:
        return M.delete(delete_ground)
    return M

def _os_local_extend_group_domain(W, groundset):
    """
    Ensure the permutation group domain contains the full ground set.

    Sage sometimes creates groups whose domain is a strict subset of the ground
    set (e.g. after stabilisers).  We extend the domain by adding missing points
    as fixed points so that evaluating permutations on ground elements is safe.
    """

    dom = set(W.domain())
    target = dom.union(set(groundset))
    if target == dom:
        return W
    return PermutationGroup(W.gens(), domain=sorted(target))

def _os_local_require_ground_action(M, W):
    """
    Guard against passing a flats-action group into OS-local routines.

    OS-local expects a group acting on the *ground set*.  A common mistake is to
    pass `induced_action_on_flats_group(M, ...)`, whose domain consists of flats
    (typically `frozenset`s).  This check raises a clearer error in that case.
    """

    ground = set(M.groundset())
    dom = set(W.domain())
    if ground.issubset(dom):
        return
    if dom and all(isinstance(x, frozenset) for x in dom) and any(
        not isinstance(e, frozenset) for e in ground
    ):
        raise RuntimeError(
            "OS-local requires a group acting on the ground set; "
            "pass M.automorphism_group() (or a subgroup on the same domain), "
            "not the induced action on flats from induced_action_on_flats_group(M, ...)."
        )

def os_local_graded_characters(M, W=None):
    """
    Graded characters of the local Orlik–Solomon algebra of ``M``.

    Returns a list `[χ_0, χ_1, ..., χ_r]` where `χ_i` is the character of the
    degree-i piece of `OS(M; QQ)` as a class function on `W`.

    Notes
    -----
    This routine is relatively slow: it computes traces by acting on a basis of
    the OS algebra.  Use OS-global for the default characteristic dispatcher.
    """

    if W is None:
        W = M.automorphism_group()
    _os_local_require_ground_action(M, W)
    W = _os_local_extend_group_domain(W, M.groundset())

    A = M.orlik_solomon_algebra(QQ)
    basis = A.basis()
    rank = M.rank()
    groundset = list(M.groundset())

    degree_keys = {
        i: [key for key in basis.keys() if len(key) == i]
        for i in range(rank + 1)
    }

    unit = A.one()
    deg1 = {
        e: A.subset_image(frozenset({e}))
        for e in groundset
    }

    def degree_character(i):
        keys = degree_keys[i]
        if not keys:
            return ClassFunction(W, [0] * len(W.conjugacy_classes()))

        values = []
        for conj_class in W.conjugacy_classes():
            g = conj_class.representative()

            trace = QQ(0)
            for key in keys:
                seq = sorted(key)
                elt = unit
                for e in seq:
                    img_e = g(e)
                    elt = elt * deg1[img_e]

                coeff_diag = QQ(0)
                for monomial, coeff in zip(
                    elt.monomials(), elt.coefficients()
                ):
                    support = monomial.leading_support()
                    if support == key:
                        coeff_diag += QQ(coeff)
                trace += coeff_diag
            values.append(trace)

        return ClassFunction(W, values)

    return [degree_character(i) for i in range(rank + 1)]

def equivariant_characteristic_os_local(M, W=None):
    """
    Equivariant characteristic polynomial via local OS^{top}(M_F) data. (Method 2)
    """
    if W is None:
        W = M.automorphism_group()

    original_W = W
    _os_local_require_ground_action(M, W)
    W = _os_local_extend_group_domain(W, M.groundset())

    P = M.lattice_of_flats()
    r = M.rank()

    # Convention: if M has a loop then χ_M(t) = 0 (Sage's convention).
    # Enforce this explicitly so OS-local/OS-global agree on all matroids.
    try:
        if len(M.loops()) > 0:
            cf0 = ClassFunction(W, [0] * len(W.conjugacy_classes()))
            coeffs = [cf0 for _ in range(r + 1)]
            if W != original_W:
                reps = [c.representative() for c in original_W.conjugacy_classes()]
                restored = []
                for cf in coeffs:
                    vals = [cf(W(g)) for g in reps]
                    restored.append(ClassFunction(original_W, vals))
                return restored
            return coeffs
    except Exception:
        pass

    if r == 0:
        # Rank-0 matroid:
        # - If the ground set is empty, χ_M(t) = 1.
        # - If the ground set is nonempty, M has loops and χ_M(t) = 0.
        is_empty = (len(M.groundset()) == 0)
        cf0 = (
            ClassFunction(W, [1] * len(W.conjugacy_classes()))
            if is_empty
            else ClassFunction(W, [0] * len(W.conjugacy_classes()))
        )
        if W != original_W:
            reps = [c.representative() for c in original_W.conjugacy_classes()]
            vals = [cf0(W(g)) for g in reps]
            return [ClassFunction(original_W, vals)]
        return [cf0]

    coeffs = [
        ClassFunction(W, [0] * len(W.conjugacy_classes()))
        for _ in range(r + 1)
    ]

    for F in P:
        rk_F = P.rank(F)
        crk_F = r - rk_F

        W_domain = set(W.domain())
        F_in_domain = [x for x in F if x in W_domain]
        if F_in_domain:
            stabiliser = W.stabilizer(frozenset(F_in_domain), "OnSets")
        else:
            stabiliser = W

        M_F = _os_local_localisation_at_flat(M, F)

        os_chars_F = os_local_graded_characters(M_F, stabiliser)
        top_deg = M_F.rank()
        os_top = os_chars_F[top_deg]

        induced = os_top.induct(W)
        weight = QQ(stabiliser.order()) / QQ(W.order())
        coeffs[crk_F] += weight * ((-1) ** rk_F) * induced

    if W != original_W:
        reps = [c.representative() for c in original_W.conjugacy_classes()]
        restored_coeffs = []
        for cf in coeffs:
            vals = [cf(W(g)) for g in reps]
            restored_coeffs.append(ClassFunction(original_W, vals))
        return restored_coeffs

    return coeffs


# =============================================================================
# PART 3: Global Orlik–Solomon (Wedge)
# =============================================================================

def _os_global_require_ground_action(M, W):
    """
    Guard against passing a flats-action group into OS-global routines.
    """

    ground = set(M.groundset())
    dom = set(W.domain())
    if dom and all(isinstance(x, frozenset) for x in dom) and any(not isinstance(e, frozenset) for e in ground):
        raise RuntimeError(
            "OS-global requires a group acting on the ground set; "
            "pass M.automorphism_group() (or a subgroup on the same domain), "
            "not the induced action on flats from induced_action_on_flats_group(M, ...)."
        )

def os_global_graded_characters(M, W=None):
    """
    Graded characters of the global Orlik–Solomon algebra of ``M``.

    Returns a list `[χ_0, χ_1, ..., χ_r]` where `χ_i` is the character of the
    degree-i piece of `OS(M; QQ)` as a class function on `W`.
    """

    if W is None:
        W = M.automorphism_group()
    _os_global_require_ground_action(M, W)

    A = M.orlik_solomon_algebra(QQ)
    basis = A.basis()
    rank = M.rank()
    groundset = list(M.groundset())

    degree_keys = {
        i: [key for key in basis.keys() if len(key) == i]
        for i in range(rank + 1)
    }

    unit = A.one()
    deg1 = {
        e: A.subset_image(frozenset({e}))
        for e in groundset
    }

    def degree_character(i):
        keys = degree_keys[i]
        if not keys:
            return ClassFunction(W, [0] * len(W.conjugacy_classes()))

        if i == 0:
            return ClassFunction(W, [1] * len(W.conjugacy_classes()))

        values = []
        for conj_class in W.conjugacy_classes():
            g = conj_class.representative()
            trace = QQ(0)
            for key in keys:
                seq = sorted(key)
                elt = unit
                for e in seq:
                    img_e = g(e)
                    elt = elt * deg1[img_e]

                coeff_diag = QQ(0)
                for monomial, coeff in zip(elt.monomials(), elt.coefficients()):
                    support = monomial.leading_support()
                    if support == key:
                        coeff_diag += QQ(coeff)
                trace += coeff_diag
            values.append(trace)
        return ClassFunction(W, values)

    return [degree_character(i) for i in range(rank + 1)]

def equivariant_characteristic_os_global(M, W=None):
    """
    Equivariant characteristic polynomial via the global OS wedge formula. (Method 3)
    """
    if W is None:
        W = M.automorphism_group()
    _os_global_require_ground_action(M, W)

    r = M.rank()

    # Convention: if M has a loop then χ_M(t) = 0 (Sage's convention).
    # The raw global OS graded character does not automatically encode this.
    try:
        if len(M.loops()) > 0:
            cf0 = ClassFunction(W, [0] * len(W.conjugacy_classes()))
            return [cf0 for _ in range(r + 1)]
    except Exception:
        pass

    os_chars = os_global_graded_characters(M, W)
    if len(os_chars) != r + 1:
        raise RuntimeError(f"os_global_graded_characters returned unexpected length {len(os_chars)}")

    if r == 0:
        return [os_chars[0]]

    coeffs = []
    for k in range(r + 1):
        idx = r - k
        chi_idx = os_chars[idx]
        coeffs.append(((-1) ** (r - k)) * chi_idx)

    return coeffs
