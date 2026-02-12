"""
check_bn_kl.sage
----------------

Consistency checker for the two B_n thagomizer formulas in the README/paper:

1) P_{T_n}^{B_n}(t)
2) Q_{T_n}^{B_n}(t)

The checker compares the expected bipartition multiplicities with the output of
`analyse_thagomizer(..., compute_inverse=True)` coefficient-by-coefficient.

Typical usage::

    sage -c "load('check_bn_kl.sage'); check_bn_kl(1, 4)"
"""

load("families/tn.sage")


def _expected_p_bn(n):
    data = {0: {((n,), ()): 1}}
    for k in range(1, n // 2 + 1):
        coeff = {}
        for i in range(2 * k, n + 1):
            lam = () if n - i == 0 else (n - i,)
            mu = (i - 2 * k + 2,) + (2,) * (k - 1)
            coeff[(lam, mu)] = 1
        data[k] = coeff
    return data


def _expected_q_bn(n):
    data = {}
    for k in range(0, n // 2 + 1):
        coeff = {}
        for i in range(2 * k, n + 1):
            lam = (1,) * (n - i)
            mu = (2,) * k + (1,) * (i - 2 * k)
            coeff[(lam, mu)] = 1
        data[k] = coeff
    return data


def _to_mult_dict(decomposition):
    return {(tuple(lam), tuple(mu)): int(mult) for (lam, mu), mult in decomposition}


def check_bn_kl(n_min=1, n_max=4, *, verbose=True):
    """
    Verify B_n formulas for `P_{T_n}^{B_n}(t)` and `Q_{T_n}^{B_n}(t)`.

    Parameters
    ----------
    n_min, n_max : int
        Inclusive range of n values to check.
    verbose : bool
        Print per-degree mismatch details when a failure occurs.
    """

    n_min = ZZ(n_min)
    n_max = ZZ(n_max)
    if n_min < 0 or n_max < 0:
        raise ValueError("n_min and n_max must be nonnegative.")
    if n_min > n_max:
        raise ValueError("Require n_min <= n_max.")

    overall_ok = True
    results = {}

    for n in range(n_min, n_max + 1):
        out = analyse_thagomizer(
            n, verbose=False, return_data=True, compute_inverse=True
        )
        p_expected = _expected_p_bn(n)
        q_expected = _expected_q_bn(n)

        p_ok = True
        q_ok = True

        for k, res in enumerate(out["decomposition"]):
            got = _to_mult_dict(res["decomposition"])
            exp = p_expected.get(k, {})
            if got != exp:
                p_ok = False
                if verbose:
                    print(f"P mismatch at n={n}, k={k}")
                    print(f"  got: {got}")
                    print(f"  exp: {exp}")

        for k, res in enumerate(out["inverse_decomposition"]):
            got = _to_mult_dict(res["decomposition"])
            exp = q_expected.get(k, {})
            if got != exp:
                q_ok = False
                if verbose:
                    print(f"Q mismatch at n={n}, k={k}")
                    print(f"  got: {got}")
                    print(f"  exp: {exp}")

        ok = p_ok and q_ok
        overall_ok = overall_ok and ok
        results[int(n)] = {"P_ok": p_ok, "Q_ok": q_ok}
        print(f"n={n}: P={'OK' if p_ok else 'FAIL'}, Q={'OK' if q_ok else 'FAIL'}")

    print(f"overall: {'OK' if overall_ok else 'FAIL'}")
    return {"ok": overall_ok, "results": results}

