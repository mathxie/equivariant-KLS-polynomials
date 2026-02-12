"""
ekl.sage
~~~~~~~~

Core entry points for equivariant Kazhdan-Lusztig computations:

- ``ekl(P_or_M, W=None)`` returns equivariant KL coefficients ``[C_0, C_1, ...]``.
- ``equivariant_inverse_kl(P_or_M, W=None)`` (alias ``ikl``) returns inverse
  coefficients ``[Q_0, Q_1, ...]`` in the incidence-algebra sense.

Dispatch rules:

- If input is a matroid ``M``:
  - ``W is None``: use ``Aut(M)`` and return class functions on it.
  - ``W`` provided: treat it as a ground-set action, induce to flats
    internally, and inflate results back to the provided group.
- If input is a bounded poset ``P``:
  - ``W is None``: use ``Aut(P)`` (Hasse-diagram automorphisms).
  - ``W`` provided: treat it as the given poset action.

This file explicitly loads ``core/equivariant_characteristic.sage`` so shared
helpers for eKL/IKL/mu/characteristic/Z(Y) are available.

Examples::

    HOME= SAGE_CACHE_DIR=.sage_cache sage -c "load('core/ekl.sage'); load('families/tn.sage'); analyse_thagomizer(3, verbose=False)"
    HOME= SAGE_CACHE_DIR=.sage_cache sage -c "load('core/ekl.sage'); load('families/wheel.sage'); analyse_wheel(4, compute_inverse=True, compute_mobius=True, compute_characteristic=True, verbose=False)"
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
    load(os.path.join(_CORE_DIR, "equivariant_characteristic.sage"))
else:
    load("core/equivariant_characteristic.sage")


# Global cache for canonicalized (Poset, Group) coefficient data.
# Recursive calls revisit the same canonical instances; caching avoids repeated GAP/Sage work.
DD = {}


########################################################################
# Coefficient helper
########################################################################
def coeff(poly_coeffs, group, idx):
    """
    Safely read the idx-th coefficient from a list.
    Return the zero class function when idx is out of range.
    """

    if 0 <= idx < len(poly_coeffs):
        return poly_coeffs[idx]
    return ClassFunction(group, [0] * len(group.conjugacy_classes()))


########################################################################
# Recursive core: canonicalized (P_can, W_can)
########################################################################
def _compute_equivariant_KL(P_can, W_can):
    """
    Compute equivariant KL coefficients for canonicalized (P_can, W_can).
    """

    if not P_can.is_bounded():
        raise ValueError("Poset must have unique bottom and top elements.")

    n = P_can.rank()
    if n < 3:
        return [ClassFunction(W_can, [1] * len(W_can.conjugacy_classes()))]

    orbit_reps = {orb[0] for orb in W_can.orbits()}.intersection(Set(P_can))
    temp = [
        ClassFunction(W_can, [0] * len(W_can.conjugacy_classes()))
        for _ in range(n + 1)
    ]

    for F in orbit_reps - {P_can.bottom()}:
        stab = W_can.stabilizer(F)
        upper = P_can.subposet(P_can.interval(F, P_can.top()))
        rank_F = P_can.rank(F)
        upper_coeffs = ekl(upper, stab)
        for deg, cf in enumerate(upper_coeffs):
            temp[rank_F + deg] += cf.induct(W_can)

    res = [
        ClassFunction(W_can, [0] * len(W_can.conjugacy_classes()))
        for _ in range(n + 1)
    ]
    for i in range(n + 1):
        res[i] = temp[n - i] - temp[i]
    return res[: (n + 1) // 2]

########################################################################
# Public wrapper: ekl(P, W) / ekl(M)
########################################################################

def _ekl_poset(P, W=None):
    """
    Compute equivariant KL coefficients on a poset P with a group action W.
    """

    if W is None:
        W = P.hasse_diagram().automorphism_group()

    # Step 1: shrink the group action support to effective points.
    W_small, pi = shrink_group_to_poset(W, P)

    # Step 2: canonicalize to obtain stable cache keys.
    P_can, W_can, relabel, phi = canonicalize_action(P, W_small)

    # Step 3: cache canonical instance coefficients.
    key = (P_can, W_can)
    if key in DD:
        coeffs_can = DD[key]
    else:
        coeffs_can = _compute_equivariant_KL(P_can, W_can)
        # Cache miss diagnostics are noisy in batch runs; opt in via env var.
        if os.environ.get("EKL_PRINT_NEW", "") not in ("", "0"):
            print("new computation", P_can)
        DD[key] = coeffs_can

    # Step 4: inflate back to the original group W.
    result = []
    for cf_can in coeffs_can:
        cf_small = inflate_classfunction(cf_can, phi)  # W_can → W_small
        cf_big = inflate_classfunction(cf_small, pi)  # W_small → W
        assert cf_big.domain() == W
        result.append(cf_big)

    # Trivial-group corner case: return constant-one class function.
    if len(W) == 1 and len(result) == 1:
        return [ClassFunction(W, [1] * len(W.conjugacy_classes()))]
    return result


def _induced_action_on_flats_hom(G, W):
    """
    Build the homomorphism pi: G -> W induced by the action on flats.
    """

    flats = list(W.domain())
    images = [
        [frozenset(sigma(e) for e in F) for F in flats]
        for sigma in G.gens()
    ]
    return G.hom(images, W)


def induced_action_on_flats(G, W):
    """
    Public wrapper: homomorphism pi: G -> W induced by the action on flats.

    Many scripts historically imported this from `families/tn.sage`; we expose
    it here so workflows that only need the pullback/inflation hom can depend
    solely on core modules.
    """

    return _induced_action_on_flats_hom(G, W)


def _coerce_group_action_on_domain(W, target_domain, *, sort_target=False):
    """
    Ensure a permutation group can act on a target domain by fixing missing points.

    This is used for *user-provided* actions where ``W.domain()`` may be a
    strict subset of the target domain. If the target domain is already
    contained in ``W.domain()``, we return ``W`` unchanged to preserve
    subgroup embeddings needed by induction inside the eKL recursion.

    Returns
    -------
    (W_act, inj)
        - W_act: permutation group acting on the target domain.
        - inj : homomorphism W -> W_act if we had to extend the action;
                otherwise None.
    """

    try:
        domain = list(W.domain())
    except Exception as exc:
        raise ValueError("W must be a permutation group with a domain.") from exc

    target = list(target_domain)
    if sort_target:
        try:
            target = sorted(target)
        except Exception:
            target = sorted(target, key=lambda e: repr(e))

    target_set = set(target)
    domain_set = set(domain)

    # If W already acts on all target points, keep it unchanged to preserve
    # subgroup structure (critical for cf.induct(supergroup)).
    if target_set.issubset(domain_set):
        for sigma in W.gens():
            for e in target:
                try:
                    img = sigma(e)
                except Exception as exc:
                    raise ValueError("W generator cannot act on target domain.") from exc
                if img not in target_set:
                    raise ValueError("W moves an element outside the target domain.")
        return W, None

    # Otherwise, extend the action by fixing points outside W.domain().
    images = []
    for sigma in W.gens():
        image = []
        for e in target:
            if e in domain_set:
                try:
                    img = sigma(e)
                except Exception as exc:
                    raise ValueError("W generator cannot act on target domain.") from exc
            else:
                img = e
            if img not in target_set:
                raise ValueError("W moves an element outside the target domain.")
            image.append(img)
        images.append(image)

    W_act = PermutationGroup(gens=images, domain=target)
    images_as_elements = [W_act(g_img) for g_img in images]
    inj = W.hom(images_as_elements, W_act)
    return W_act, inj


def coerce_group_action_on_domain(W, target_domain, *, sort_target=False):
    """
    Public wrapper around ``_coerce_group_action_on_domain``.

    This is useful when you want to pass a group whose domain is smaller or
    larger than the combinatorial object (matroid ground set / poset elements):
    missing points are treated as fixed points.
    """

    return _coerce_group_action_on_domain(W, target_domain, sort_target=sort_target)


def _validate_group_action_on_matroid(M, G):
    """
    Validate that G acts on the ground set and preserves the matroid.
    """

    ground = list(M.groundset())
    try:
        ground = sorted(ground)
    except Exception:
        ground = sorted(ground, key=lambda e: repr(e))

    G_act, inj = _coerce_group_action_on_domain(G, ground)

    for sigma in G_act.gens():
        mapping = {e: sigma(e) for e in ground}
        if not M.is_isomorphism(M, mapping):
            raise ValueError("G generator is not a matroid automorphism.")

    return G_act, inj


def validate_group_action_on_matroid(M, G):
    """Public wrapper around ``_validate_group_action_on_matroid``."""

    return _validate_group_action_on_matroid(M, G)


def _validate_group_action_on_poset(P, W):
    """
    Validate that W acts on the poset elements and preserves the order.
    """

    elements = list(P.list())
    W_act, inj = _coerce_group_action_on_domain(W, elements)

    cover_edges = P.hasse_diagram().edges(labels=False)
    for sigma in W_act.gens():
        for x, y in cover_edges:
            if not P.le(sigma(x), sigma(y)):
                raise ValueError("W generator does not preserve the poset order.")

    return W_act, inj


def validate_group_action_on_poset(P, W):
    """Public wrapper around ``_validate_group_action_on_poset``."""

    return _validate_group_action_on_poset(P, W)


def ekl(P, W=None):
    """
    Coefficient list of the equivariant Kazhdan-Lusztig polynomial.

    For a bounded poset P of rank r with action W, the equivariant KL polynomial is

        P_P^W(t) = sum_{i=0}^{floor(r/2)} C_i t^i,

    where C_i lies in the representation ring R(W), implemented as a class function on W.
    This function returns [C_0, ..., C_floor(r/2)] of length floor(r/2)+1.

    If input is a matroid M, we use its lattice of flats L(M) as P.
    - If W is omitted, use the induced action of Aut(M) on flats and return coefficients on Aut(M).
    - If W is provided, treat it as a ground-set action and return coefficients on W.
      If W domain differs from the ground set, it is coerced automatically.
      To use an action directly on L(M), call ``ekl(M.lattice_of_flats(), W)`` explicitly.

    INPUT:

    - ``P`` -- a bounded poset, or a matroid (we use its lattice of flats).
    - ``W`` -- (default: ``None``) a permutation group acting on ``P`` (poset action)
      or on the matroid ground set (ground-set action).

    OUTPUT:

    A list of class functions ``[C_0, C_1, ...]`` on the chosen group, ordered
    by increasing degree, with ``deg(P_P^W) <= floor(rk(P)/2)``.

    EXAMPLES::

        sage: load('core/ekl.sage')
        sage: M = matroids.Uniform(3, 5)
        sage: coeffs = ekl(M)
        sage: G = M.automorphism_group()
        sage: [cf(G.identity()) for cf in coeffs]
        [1, 5]
    """

    if hasattr(P, "lattice_of_flats"):
        M = P
        if W is None:
            L = M.lattice_of_flats()
            G = M.automorphism_group()
            W = induced_action_on_flats_group(M, G, flats=L)
            pi = _induced_action_on_flats_hom(G, W)
            coeffs = _ekl_poset(L, W)
            return [inflate_classfunction(cf, pi, G, W) for cf in coeffs]
        W_act, inj = _validate_group_action_on_matroid(M, W)
        L = M.lattice_of_flats()
        W_flats = induced_action_on_flats_group(M, W_act, flats=L)
        pi = _induced_action_on_flats_hom(W_act, W_flats)
        coeffs = _ekl_poset(L, W_flats)
        lifted = [inflate_classfunction(cf, pi, W_act, W_flats) for cf in coeffs]
        if inj is None:
            return lifted
        return [inflate_classfunction(cf, inj, W, W_act) for cf in lifted]
        P = L
    if W is not None:
        W_act, inj = _validate_group_action_on_poset(P, W)
        coeffs = _ekl_poset(P, W_act)
        if inj is None:
            return coeffs
        return [inflate_classfunction(cf, inj, W, W_act) for cf in coeffs]

    return _ekl_poset(P, W)


########################################################################
# Inverse equivariant Kazhdan-Lusztig (inverse eKL / IKL)
########################################################################

# Cache for inverse KL on canonical (poset, group) pairs.
_IKL_CACHE = {}


def _zero_cf(group):
    """
    Zero class function on ``group``.

    This is a tiny wrapper around ``zero_classfunction`` (from `core/bases.sage`)
    kept to make the inverse-KL recursion formulas read closer to the paper.
    """

    return zero_classfunction(group)


def _add_cf(cf1, cf2):
    """
    Pointwise sum of two class functions on the same group.

    Notes
    -----
    We build the result by evaluating on conjugacy-class representatives to
    avoid parent/coercion issues (similar in spirit to
    ``classfunction_pointwise_product``).
    """

    group = cf1.domain()
    reps = group.conjugacy_classes_representatives()
    return ClassFunction(group, [cf1(g) + cf2(g) for g in reps])


def _neg_cf(cf):
    """
    Pointwise negation of a class function.
    """

    group = cf.domain()
    reps = group.conjugacy_classes_representatives()
    return ClassFunction(group, [-cf(g) for g in reps])


def _mul_cf(cf1, cf2):
    """
    Multiply class functions by values on conjugacy class representatives.

    We avoid using ``cf1 * cf2`` because Sage/GAP may attach different
    character-table objects to class functions even when the underlying
    group is the same, which can trigger “no product of class functions of
    different tables”.
    """

    return classfunction_pointwise_product(cf1, cf2)


def _ikl_poset(P, W=None):
    """
    Compute inverse eKL coefficients on a poset P with a group action W.
    """

    if W is None:
        W = P.hasse_diagram().automorphism_group()

    # Shrink group to the actual poset domain.
    W_small, pi = shrink_group_to_poset(W, P)

    # Canonicalise (P, W_small) for caching.
    P_can, W_can, relabel, phi = canonicalize_action(P, W_small)

    # Compute canonical Q.
    Q_can = _equivariant_inverse_kl_canonical(P_can, W_can)

    # Inflate back to the original group W via W_can -> W_small -> W.
    result = []
    for cf_can in Q_can:
        cf_small = inflate_classfunction(cf_can, phi)  # W_can → W_small
        cf_big = inflate_classfunction(cf_small, pi)   # W_small → W
        result.append(cf_big)
    return result


def _equivariant_inverse_kl_canonical(P_can, W_can):
    """
    Compute Q_{P_can}^{W_can}(t) on a canonical (poset, group) pair.

    Returns a list of class functions [Q0, Q1, ...] on W_can, ordered by
    increasing t-degree, with deg(Q) < rk(P_can)/2.
    """

    key = (P_can, W_can)
    cached = _IKL_CACHE.get(key)
    if cached is not None:
        return cached

    r = P_can.rank()
    # Base case: rank 0, Q(t) = 1 in degree 0.
    if r == 0:
        cf = ClassFunction(W_can, [1] * len(W_can.conjugacy_classes()))
        _IKL_CACHE[key] = [cf]
        return [cf]

    # Maximum allowed degree: strictly less than r/2.
    max_deg = (r - 1) // 2

    # We work with coefficients C_k on the right-hand side of
    #
    #   (-1)^r ( t^r Q(t^{-1}) - Q(t) )
    #     = sum_{[F]≠[top]} (-1)^{rk(F)} Ind_{W_F}^W( Q_{down(F)} ⊗ t^{rk(up(F))} H_{up(F)}(t^{-1}) ).
    #
    # For 0 ≤ k ≤ max_deg we then have q_k = (-1)^{r+1} C_k.
    C = [_zero_cf(W_can) for _ in range(r + 1)]

    top = P_can.top()

    # Orbit representatives of W_can acting on elements of P_can.
    orbit_reps = {orb[0] for orb in W_can.orbits()}.intersection(Set(P_can))

    for F in orbit_reps:
        if F == top:
            # The F = top term is the one containing Q_P itself; it is moved
            # to the left-hand side, so we skip it here.
            continue

        # Stabiliser subgroup W_F.
        W_F = W_can.stabilizer(F)

        # Lower interval [bottom, F] and upper interval [F, top].
        elems_down = [x for x in P_can if P_can.le(x, F)]
        P_down = P_can.subposet(elems_down)

        elems_up = [x for x in P_can if P_can.le(F, x)]
        P_up = P_can.subposet(elems_up)
        r_up = P_up.rank()

        # Recursive Q on the lower interval, with group W_F.
        Q_down = equivariant_inverse_kl(P_down, W_F)

        # Equivariant characteristic polynomial on the upper interval.
        H_up = equivariant_characteristic_polynomial(P_up, W_F)

        # Build the polynomial Q_down(t) ⊗ t^{r_up} H_up(t^{-1}) on W_F.
        local = {}
        for i, q_cf in enumerate(Q_down):
            for j, h_cf in enumerate(H_up):
                d = i + r_up - j
                if d < 0 or d > r:
                    continue
                prod = _mul_cf(q_cf, h_cf)
                if d in local:
                    local[d] = _add_cf(local[d], prod)
                else:
                    local[d] = prod

        sign = (-1) ** P_can.rank(F)
        for d, cf_loc in local.items():
            # Induce from W_F to W_can.
            cf_ind = cf_loc.induct(W_can)
            if sign == -1:
                cf_ind = _neg_cf(cf_ind)
            C[d] = _add_cf(C[d], cf_ind)

    # Now extract the small-degree coefficients of Q from C.
    Q_can = []
    for k in range(max_deg + 1):
        coeff = C[k]
        # q_k = (-1)^{r+1} C_k
        if (r + 1) % 2 == 1:
            coeff = _neg_cf(coeff)
        Q_can.append(coeff)

    _IKL_CACHE[key] = Q_can
    return Q_can


def equivariant_inverse_kl(P, W=None, *, return_polynomial=False):
    """
    Genuine equivariant inverse Kazhdan–Lusztig polynomial Q^W(P; t).

    This is the (incidence-algebra) inverse of the equivariant KL polynomial in
    the sense of Proudfoot / Gao–Li–Xie.

    Input
    -----
    P : Poset or matroid
        If P is a matroid, its lattice of flats is used.  If P is already a
        bounded poset, it is used directly.
    W : PermutationGroup (optional)
        - If P is a bounded poset: finite group acting on P by poset automorphisms.
        - If P is a matroid: a group acting on the *ground set* and preserving the
          matroid (domain may be smaller/larger; missing points are fixed).
          If you want a group acting on the lattice of flats, call
          ``equivariant_inverse_kl(P.lattice_of_flats(), W)`` explicitly.

    Output
    ------
    A list of class functions [Q0, Q1, ...] on the original group W, ordered
    by increasing t-degree, with deg(Q) < rk(P)/2.

    EXAMPLES::

        sage: load('core/ekl.sage')
        sage: M = matroids.Uniform(3, 5)
        sage: Q = equivariant_inverse_kl(M)  # same as ikl(M)
        sage: G = M.automorphism_group()
        sage: [cf(G.identity()) for cf in Q]
        [6, 5]
    """

    if hasattr(P, "lattice_of_flats"):
        M = P
        L = M.lattice_of_flats()

        if W is None:
            G = M.automorphism_group()
            W_flats = induced_action_on_flats_group(M, G)
            pi = _induced_action_on_flats_hom(G, W_flats)
            coeffs = _ikl_poset(L, W_flats)
            lifted = [inflate_classfunction(cf, pi, G, W_flats) for cf in coeffs]
            return lifted if return_polynomial else lifted

        W_act, inj = _validate_group_action_on_matroid(M, W)
        W_flats = induced_action_on_flats_group(M, W_act)
        pi = _induced_action_on_flats_hom(W_act, W_flats)
        coeffs = _ikl_poset(L, W_flats)
        lifted = [inflate_classfunction(cf, pi, W_act, W_flats) for cf in coeffs]
        if inj is None:
            return lifted if return_polynomial else lifted
        pulled = [inflate_classfunction(cf, inj, W, W_act) for cf in lifted]
        return pulled if return_polynomial else pulled

    if W is not None:
        W_act, inj = _validate_group_action_on_poset(P, W)
        coeffs = _ikl_poset(P, W_act)
        if inj is None:
            return coeffs if return_polynomial else coeffs
        pulled = [inflate_classfunction(cf, inj, W, W_act) for cf in coeffs]
        return pulled if return_polynomial else pulled

    coeffs = _ikl_poset(P, W)
    return coeffs if return_polynomial else coeffs


def ikl(P, W=None, *, return_polynomial=False):
    """
    Alias for ``equivariant_inverse_kl``.

    This mirrors the `ekl(P, W)` entry point:

    - If ``P`` is a matroid, then by default we return class functions on
      ``Aut(P)`` (the ground-set action), not on the lattice-of-flats group.
    - If ``P`` is a bounded poset, then by default we return class functions on
      ``Aut(P)`` (poset automorphisms).

    EXAMPLES::

        sage: load('core/ekl.sage')
        sage: M = matroids.Uniform(3, 5)
        sage: Q = ikl(M)
        sage: G = M.automorphism_group()
        sage: [cf(G.identity()) for cf in Q]
        [6, 5]
    """

    return equivariant_inverse_kl(P, W, return_polynomial=return_polynomial)


# Short aliases matching common notation.
iekl = ikl
eikl = ikl
