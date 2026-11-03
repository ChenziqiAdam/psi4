#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2026 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# @END LICENSE
#
"""Scientific sanitizers (observation only).

Each check reads state already present at its observation point (or re-evaluates a pure function on private
copies), appends a JSON line to the file named by ``PSI4_SANITIZE_LOG`` when its alarm predicate holds, and never
changes a return value, raises, or touches persistent state. With the variable unset every hook is a no-op.
A failure inside a checker is written to ``<log>.err`` and otherwise swallowed. Predicates are module-level
functions on plain arrays so that they can be tested on synthetic observed states.
"""
import contextlib
import io
import json
import math
import os

import numpy as np

_BUSY = False  # re-entrancy guard for checks that re-call the instrumented function

# physical constants for the independent paths (CODATA 2018, SI exact where defined)
_H = 6.62607015e-34
_C = 299792458.0
_AMU = 1.66053906660e-27
_A0 = 5.29177210903e-11


# ----------------------------------------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------------------------------------
def enabled():
    return bool(os.environ.get("PSI4_SANITIZE_LOG"))


def _append(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def alarm(sid, **detail):
    path = os.environ.get("PSI4_SANITIZE_LOG")
    if not path:
        return
    rec = {"id": sid, "pid": os.getpid(), "detail": {k: _jsonable(v) for k, v in detail.items()}}
    key = os.environ.get("PSI4_SANITIZE_KEY")
    if key:
        rec["key"] = key
    _append(path, rec)


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def guarded(fn):
    """Run a hook; swallow and record any exception so the host calculation is unaffected."""

    def wrapper(*args, **kwargs):
        if not enabled():
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- instrumentation must never disturb the calculation
            try:
                _append(os.environ["PSI4_SANITIZE_LOG"] + ".err", {"hook": fn.__name__, "error": repr(exc)})
            except Exception:  # noqa: BLE001
                pass
            return None

    wrapper.__name__ = fn.__name__
    return wrapper


# ----------------------------------------------------------------------------------------------------------
# pure predicates (each returns None when the law holds, else a dict of evidence)
# ----------------------------------------------------------------------------------------------------------
def pred_net_force(g, tol):
    """P4-GRD-001: forces sum to zero per axis."""
    s = np.asarray(g, dtype=float).sum(axis=0)
    return {"net_force": s, "tol": tol} if np.max(np.abs(s)) > tol else None


def pred_net_torque(g, r, tol):
    """P4-GRD-002: net torque about the origin vanishes (net force vanishes by P4-GRD-001)."""
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    t = np.cross(r, g).sum(axis=0)
    scale = max(1.0, float(np.max(np.linalg.norm(r, axis=1))))
    return {"net_torque": t, "tol": tol * scale} if np.max(np.abs(t)) > tol * scale else None


def pred_hessian_symmetry(H, tol):
    H = np.asarray(H, dtype=float)
    d = float(np.max(np.abs(H - H.T)))
    return {"max_asym": d, "tol": tol} if d > tol else None


def _trans_vectors(nat):
    T = np.zeros((3, 3 * nat))
    for k in range(3):
        T[k, k::3] = 1.0
    return T


def pred_acoustic(H, nat, tol):
    """P4-HES-002: H t_k = 0 for the three unit translations."""
    H = np.asarray(H, dtype=float)
    res = H @ _trans_vectors(nat).T
    d = float(np.max(np.abs(res)))
    return {"max_residual": d, "tol": tol} if d > tol else None


def pred_rotational(H, g, r, tol):
    """P4-HES-003: sum_B H_AB (n x r_B) = n x g_A for the three axes n."""
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    worst = 0.0
    for k in range(3):
        n = np.zeros(3)
        n[k] = 1.0
        u = np.cross(n, r).reshape(-1)
        rhs = np.cross(n, g).reshape(-1)
        worst = max(worst, float(np.max(np.abs(H @ u - rhs))))
    return {"max_residual": worst, "tol": tol} if worst > tol else None


def pred_dipole_derivative_sum(dmu, charge, tol):
    """P4-HES-004: sum_A d mu_i / d R_Aj = Q delta_ij.  `dmu` has shape (3 nat, 3): row = Cartesian displacement."""
    d = np.asarray(dmu, dtype=float).reshape(-1, 3, 3).sum(axis=0)  # (j, i)
    res = d - charge * np.eye(3)
    m = float(np.max(np.abs(res)))
    return {"max_residual": m, "tol": tol, "sum": d} if m > tol else None


def _signed_freq(omega):
    omega = np.asarray(omega)
    return np.sort(omega.real - np.abs(omega.imag))


def pred_mass_scaling(omega, omega_scaled, s, tol_rel=1e-8, tol_abs=1e-2):
    """P4-VIB-001: scaling all masses by s scales frequencies by s**-0.5."""
    a = _signed_freq(omega)
    b = _signed_freq(omega_scaled) * math.sqrt(s)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    tol = tol_rel * scale + tol_abs
    return {"max_diff_cm-1": d, "tol": tol} if d > tol else None


def pred_cbs_two_point(vlo, vhi, value, ratio_expected, tol=1e-8):
    """P4-CBS-001/002/003/005: (vLO - E)/(vHI - E) = ratio_expected for float or array inputs."""
    vlo = np.atleast_1d(np.asarray(vlo, dtype=float))
    vhi = np.atleast_1d(np.asarray(vhi, dtype=float))
    val = np.atleast_1d(np.asarray(value, dtype=float))
    den = vhi - val
    mask = np.abs(den) >= 1e-14
    if not np.any(mask):
        return None
    lhs = (vlo - val)[mask] / den[mask]
    err = np.abs(lhs - ratio_expected) / max(1.0, abs(ratio_expected))
    worst = float(np.max(err))
    return {"max_rel_err": worst, "ratio_expected": ratio_expected, "tol": tol} if worst > tol else None


def pred_cbs_three_point(vlo, vmd, vhi, value, tol=1e-8):
    """P4-CBS-004: residuals of one exponential at equally spaced zeta are a geometric progression."""
    vlo, vmd, vhi, val = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (vlo, vmd, vhi, value))
    a, b, c = vlo - val, vmd - val, vhi - val
    mask = np.abs(vhi) > 1e-14
    if not np.any(mask):
        return None
    lhs = b[mask] ** 2
    rhs = (a * c)[mask]
    err = np.abs(lhs - rhs) / np.maximum(1e-12, np.abs(lhs) + np.abs(rhs))
    worst = float(np.max(err))
    return {"max_rel_err": worst, "tol": tol} if worst > tol else None


def pred_cbs_trend(vlo, vhi, value):
    """P4-CBS-006: sign(E_inf - vHI) = sign(vHI - vLO)."""
    vlo = np.atleast_1d(np.asarray(vlo, dtype=float))
    vhi = np.atleast_1d(np.asarray(vhi, dtype=float))
    val = np.atleast_1d(np.asarray(value, dtype=float))
    mask = (np.abs(vhi - vlo) > 1e-12) & (np.abs(vhi) > 1e-14)
    if not np.any(mask):
        return None
    bad = np.sign((val - vhi)[mask]) != np.sign((vhi - vlo)[mask])
    return {"n_reversed": int(bad.sum()), "of": int(mask.sum())} if np.any(bad) else None


def pred_spin_density_count(D_blocks, S_blocks, n, tol):
    t = sum(float(np.trace(D @ S)) for D, S in zip(D_blocks, S_blocks))
    return {"trace": t, "n": n, "tol": tol} if abs(t - n) > tol else None


def pred_idempotency(D_blocks, S_blocks, tol):
    worst = 0.0
    for D, S in zip(D_blocks, S_blocks):
        if D.size:
            worst = max(worst, float(np.max(np.abs(D @ S @ D - D))))
    return {"max_err": worst, "tol": tol} if worst > tol else None


def pred_orthonormality(C_blocks, S_blocks, tol):
    worst = 0.0
    for C, S in zip(C_blocks, S_blocks):
        if C.size:
            worst = max(worst, float(np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1])))))
    return {"max_err": worst, "tol": tol} if worst > tol else None


def pred_fock_density_commutator(F_blocks, D_blocks, S_blocks, tol):
    sq, cnt = 0.0, 0
    for F, D, S in zip(F_blocks, D_blocks, S_blocks):
        if D.size:
            r = F @ D @ S - S @ D @ F
            sq += float(np.sum(r * r))
            cnt += r.size
    rms = math.sqrt(sq / cnt) if cnt else 0.0
    return {"rms": rms, "tol": tol} if rms > tol else None


def pred_eigen_equation(F_blocks, C_blocks, S_blocks, eps_blocks, tol):
    sq, cnt = 0.0, 0
    for F, C, S, e in zip(F_blocks, C_blocks, S_blocks, eps_blocks):
        if C.size:
            r = F @ C - S @ C @ np.diag(e)
            sq += float(np.sum(r * r))
            cnt += r.size
    rms = math.sqrt(sq / cnt) if cnt else 0.0
    return {"rms": rms, "tol": tol} if rms > tol else None


def pred_hf_energy(e_reported, enuc, H_blocks, Da_blocks, Db_blocks, Fa_blocks, Fb_blocks, tol):
    e = enuc
    for H, Da, Db, Fa, Fb in zip(H_blocks, Da_blocks, Db_blocks, Fa_blocks, Fb_blocks):
        e += 0.5 * (float(np.sum(Da * (H + Fa))) + float(np.sum(Db * (H + Fb))))
    return {"reported": e_reported, "from_matrices": e, "tol": tol} if abs(e - e_reported) > tol else None


def pred_nonpositive(value, tol, label):
    return {label: value, "tol": tol} if value > tol else None


def pred_ci_bound(e_method, e_scf, tol=1e-8):
    return {"E_method": e_method, "E_scf": e_scf} if e_method > e_scf + tol else None


def _rot_images(Z, m, xyz, tol):
    """Count proper rotations mapping the molecule onto itself. `xyz` is centred on the centre of mass."""
    n = len(Z)
    key = [(int(Z[i]), round(float(m[i]), 3)) for i in range(n)]
    norms = np.linalg.norm(xyz, axis=1)
    # reference pair of non-parallel position vectors
    a = b = None
    for i in range(n):
        if norms[i] < tol:
            continue
        for j in range(i + 1, n):
            if norms[j] < tol:
                continue
            if np.linalg.norm(np.cross(xyz[i], xyz[j])) > 1e-3 * norms[i] * norms[j] and \
                    np.linalg.norm(np.cross(xyz[i], xyz[j])) > tol * max(norms[i], norms[j]):
                a, b = i, j
                break
        if a is not None:
            break
    if a is None:
        return None  # linear or single centre
    fa = np.column_stack([xyz[a], xyz[b], np.cross(xyz[a], xyz[b])])
    found = []
    for ap in range(n):
        if key[ap] != key[a] or abs(norms[ap] - norms[a]) > tol:
            continue
        for bp in range(n):
            if bp == ap or key[bp] != key[b] or abs(norms[bp] - norms[b]) > tol:
                continue
            if abs(float(xyz[ap] @ xyz[bp]) - float(xyz[a] @ xyz[b])) > tol * max(1.0, norms[a], norms[b]) * 2:
                continue
            fb = np.column_stack([xyz[ap], xyz[bp], np.cross(xyz[ap], xyz[bp])])
            R = fb @ np.linalg.inv(fa)
            if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-2 or abs(np.linalg.det(R) - 1.0) > 1e-2:
                continue
            ok = True
            used = set()
            for i in range(n):
                p = R @ xyz[i]
                hit = None
                for j in range(n):
                    if key[j] == key[i] and np.linalg.norm(p - xyz[j]) < 2 * tol and j not in used:
                        hit = j
                        break
                if hit is None:
                    ok = False
                    break
                used.add(hit)
            if ok and not any(np.max(np.abs(R - Q)) < 5e-2 for Q in found):
                found.append(R)
    return len(found)


def independent_rotational_symmetry_number(Z, m, geom, tol=1e-2):
    Z = np.asarray(Z)
    m = np.asarray(m, dtype=float)
    geom = np.asarray(geom, dtype=float)
    com = (m[:, None] * geom).sum(axis=0) / m.sum()
    xyz = geom - com
    cnt = _rot_images(Z, m, xyz, tol)
    if cnt is not None:
        return cnt
    # linear (or single centre): inversion symmetry decides
    if len(Z) < 2:
        return None
    key = [(int(Z[i]), round(float(m[i]), 3)) for i in range(len(Z))]
    inv = all(any(key[j] == key[i] and np.linalg.norm(-xyz[i] - xyz[j]) < 2 * tol for j in range(len(Z)))
              for i in range(len(Z)))
    return 2 if inv else 1


def independent_rotational_constants(m, geom):
    """Rotational constants [cm^-1] of the non-zero principal moments, descending."""
    m = np.asarray(m, dtype=float)
    geom = np.asarray(geom, dtype=float)
    com = (m[:, None] * geom).sum(axis=0) / m.sum()
    x = geom - com
    I = np.zeros((3, 3))
    for mi, xi in zip(m, x):
        I += mi * (np.dot(xi, xi) * np.eye(3) - np.outer(xi, xi))
    lam = np.linalg.eigvalsh(I)  # amu bohr^2
    lam = lam[lam > 1e-6]
    kg_m2 = lam * _AMU * _A0 ** 2
    return np.sort(_H / (8.0 * math.pi ** 2 * _C * kg_m2) / 100.0)[::-1]


def pred_rot_symmetry_number(reported, independent):
    return {"reported": reported, "independent": independent} if reported != independent else None


def pred_rot_constants(reported, independent, tol=1e-6):
    rep = np.sort(np.asarray([x for x in reported if np.isfinite(x) and x > 1e-8], dtype=float))[::-1]
    if rep.size != independent.size:
        return {"reported": rep, "independent": independent}
    rel = np.abs(rep - independent) / independent
    return {"reported": rep, "independent": independent, "max_rel": float(rel.max())} if rel.max() > tol else None


def pred_unit_chain(c, tol=1e-8):
    """P4-CON-001: derived conversion factors agree with the fundamental ones (psi4 constants context)."""
    pairs = {
        "hartree2kJmol": (c.hartree2kJmol, c.hartree2J * c.na / 1000.0),
        "hartree2wavenumbers": (c.hartree2wavenumbers, c.hartree2J / (c.h * c.c * 100.0)),
        "bohr2angstroms": (c.bohr2angstroms, c.bohr2m * 1e10),
        "hartree2kcalmol": (c.hartree2kcalmol, c.hartree2kJmol / 4.184),
    }
    bad = {k: (a, b) for k, (a, b) in pairs.items() if abs(a - b) > tol * abs(b)}
    return bad or None


def pred_gas_constant(c, tol=1e-8):
    return {"R": c.R, "NA_kB": c.na * c.kb} if abs(c.R - c.na * c.kb) > tol * c.R else None


def pred_weight_sum(parts, tol=1e-12, label="sum"):
    s = float(sum(parts))
    return {label: s} if abs(s - 1.0) > tol else None


# ----------------------------------------------------------------------------------------------------------
# helpers on psi4 objects
# ----------------------------------------------------------------------------------------------------------
def _blocks(mat):
    return [np.array(b, dtype=float) for b in mat.nph]


def _opt(module, name, default=None):
    from psi4 import core
    try:
        return core.get_option(module, name)
    except Exception:  # noqa: BLE001
        return default


def _plain_run(wfn, molecule=None):
    """No external potential, field, solvent or EFP: the common precondition of the invariance laws."""
    from psi4 import core
    if wfn.external_pot() is not None:
        return False
    if _opt("SCF", "PERTURB_H") or _opt("SCF", "PCM") or _opt("SCF", "PE"):
        return False
    if molecule is not None and hasattr(molecule, "EFP"):
        return False
    return True


def _is_dft(wfn):
    try:
        return bool(wfn.functional().needs_xc())
    except Exception:  # noqa: BLE001
        return False


def _conv():
    d = float(_opt("SCF", "D_CONVERGENCE", 1e-6))
    r = float(_opt("CCENERGY", "R_CONVERGENCE", 1e-7))
    return d, r


# ----------------------------------------------------------------------------------------------------------
# hooks (called from the driver; each swallows its own errors)
# ----------------------------------------------------------------------------------------------------------
@guarded
def gradient_checks(wfn, molecule):
    """P4-GRD-001/002 at the end of gradient()."""
    if _is_dft(wfn) or not _plain_run(wfn, molecule):
        return
    g = np.asarray(wfn.gradient())
    if g.ndim != 2 or g.shape[1] != 3 or not np.all(np.isfinite(g)):
        return
    r = np.asarray(wfn.molecule().geometry())
    d, rr = _conv()
    tol = 1e-6 + 100.0 * max(d, rr)
    res = pred_net_force(g, tol)
    if res:
        alarm("P4-GRD-001", **res)
    res = pred_net_torque(g, r, tol)
    if res:
        alarm("P4-GRD-002", **res)


@guarded
def hessian_checks(wfn, molecule):
    """P4-HES-001..004 at the end of the analytic hessian()."""
    if _is_dft(wfn) or not _plain_run(wfn, molecule):
        return
    H = np.asarray(wfn.hessian())
    if H.ndim != 2 or not np.all(np.isfinite(H)):
        return
    nat = H.shape[0] // 3
    d, rr = _conv()
    tol = (1e-6 + 100.0 * max(d, rr)) * max(1.0, float(np.max(np.abs(H))))
    for sid, res in (("P4-HES-001", pred_hessian_symmetry(H, tol)), ("P4-HES-002", pred_acoustic(H, nat, tol))):
        if res:
            alarm(sid, **res)
    g = wfn.gradient()
    if g is not None and np.asarray(g).size == 3 * nat:
        r = np.asarray(wfn.molecule().geometry())
        res = pred_rotational(H, np.asarray(g), r, tol)
        if res:
            alarm("P4-HES-003", **res)
    dip = wfn.variables().get("CURRENT DIPOLE GRADIENT", None)
    if dip is not None:
        dmu = np.asarray(dip)
        if dmu.shape == (3 * nat, 3):
            res = pred_dipole_derivative_sum(dmu, float(wfn.molecule().molecular_charge()), 1e-5 + 100.0 * d)
            if res:
                alarm("P4-HES-004", **res)


@guarded
def vib_checks(harmonic_analysis, vibinfo, hess, geom, mass, basisset, irrep_labels, project_trans, project_rot):
    """P4-VIB-001: re-call the harmonic analysis with masses scaled by s."""
    global _BUSY
    if _BUSY:
        return
    s = 4.0
    _BUSY = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scaled, _ = harmonic_analysis(hess, geom, mass * s, basisset, irrep_labels, dipder=None,
                                          project_trans=project_trans, project_rot=project_rot)
    finally:
        _BUSY = False
    res = pred_mass_scaling(vibinfo["omega"].data, scaled["omega"].data, s)
    if res:
        alarm("P4-VIB-001", **res)


def _val(info, key):
    return float(np.asarray(info[key].data))


@guarded
def thermo_checks(thermo, filter_nonvib, constants, vibinfo, T, P, multiplicity, molecular_mass, E0, sigma,
                  rot_const, rotor_type, therminfo):
    """P4-THM-001..007 and P4-CON-001/002; re-calls thermo() on private copies with one input changed."""
    global _BUSY
    res = pred_unit_chain(constants)
    if res:
        alarm("P4-CON-001", **{k: list(v) for k, v in res.items()})
    res = pred_gas_constant(constants)
    if res:
        alarm("P4-CON-002", **res)
    if _BUSY or not (T > 0 and P > 0 and molecular_mass > 0 and sigma >= 1):
        return
    Ru = constants.R / constants.hartree2kJmol  # one unit of R in the (mEh/K) convention of thermo()

    def call(**ch):
        args = dict(T=T, P=P, multiplicity=multiplicity, molecular_mass=molecular_mass, E0=E0, sigma=sigma,
                    rot_const=rot_const, rotor_type=rotor_type)
        args.update(ch)
        with contextlib.redirect_stdout(io.StringIO()):
            return thermo(vibinfo, **args)[0]

    _BUSY = True
    try:
        h = 1e-3 * T
        up, dn = call(T=T + h), call(T=T - h)
        # THM-001/002/003 (E, H in Eh; Cv, Cp, S in mEh/K)
        cv, cp = _val(therminfo, "Cv_tot"), _val(therminfo, "Cp_tot")
        fd_cv = 1000.0 * (_val(up, "E_corr") - _val(dn, "E_corr")) / (2 * h)
        if abs(cv - fd_cv) > 1e-4 * max(abs(cv), Ru):
            alarm("P4-THM-001", Cv=cv, dE_dT=fd_cv)
        fd_cp = 1000.0 * (_val(up, "H_corr") - _val(dn, "H_corr")) / (2 * h)
        if abs(cp - fd_cp) > 1e-4 * max(abs(cp), Ru):
            alarm("P4-THM-002", Cp=cp, dH_dT=fd_cp)
        fd_s = (_val(up, "S_tot") - _val(dn, "S_tot")) / (2 * h)
        if abs(cp / T - fd_s) > 1e-4 * max(abs(cp / T), Ru / T):
            alarm("P4-THM-003", Cp_over_T=cp / T, dS_dT=fd_s)
        invariant = ["E_corr", "H_corr", "Cv_tot", "Cp_tot", "ZPE_corr"]

        def unchanged(other, keys):
            return {k: (_val(therminfo, k), _val(other, k)) for k in keys
                    if abs(_val(other, k) - _val(therminfo, k)) > 1e-12 * max(1.0, abs(_val(therminfo, k)))}

        # THM-004 pressure
        p2 = call(P=2 * P)
        dS = _val(therminfo, "S_tot") - _val(p2, "S_tot")
        bad = unchanged(p2, invariant)
        if abs(dS - Ru * math.log(2.0)) > 1e-9 * Ru or bad:
            alarm("P4-THM-004", dS=dS, expected=Ru * math.log(2.0), changed=bad)
        # THM-005 mass
        m2 = call(molecular_mass=2 * molecular_mass)
        dS = _val(m2, "S_tot") - _val(therminfo, "S_tot")
        bad = unchanged(m2, invariant + ["S_rot", "S_vib", "S_elec"])
        if abs(dS - 1.5 * Ru * math.log(2.0)) > 1e-9 * Ru or bad:
            alarm("P4-THM-005", dS=dS, expected=1.5 * Ru * math.log(2.0), changed=bad)
        # THM-006 symmetry number
        if rotor_type != "RT_ATOM":
            s2 = call(sigma=2 * sigma)
            dS = _val(therminfo, "S_tot") - _val(s2, "S_tot")
            bad = unchanged(s2, invariant + ["S_trans", "S_vib", "S_elec"])
            if abs(dS - Ru * math.log(2.0)) > 1e-9 * Ru or bad:
                alarm("P4-THM-006", dS=dS, expected=Ru * math.log(2.0), changed=bad)
    finally:
        _BUSY = False
    # THM-007 vibrational bounds (modes thermo() itself uses: real vibrations)
    vibonly = filter_nonvib(vibinfo)
    om = vibonly["omega"].data
    nvib = int(np.sum(om.imag <= om.real))
    cv_v, s_v = _val(therminfo, "Cv_vib"), _val(therminfo, "S_vib")
    zpe_v, e_v = _val(therminfo, "ZPE_vib"), _val(therminfo, "E_vib")
    thermal = e_v - zpe_v
    bad = {}
    if cv_v < -1e-12 * Ru or cv_v > nvib * Ru * (1 + 1e-12) + 1e-30:
        bad["Cv_vib"] = cv_v
    if s_v < -1e-12 * Ru:
        bad["S_vib"] = s_v
    if zpe_v < 0:
        bad["ZPE_vib"] = zpe_v
    if thermal < -1e-12 * Ru * 0.001 * T or thermal > nvib * Ru * 0.001 * T * (1 + 1e-12) + 1e-30:
        bad["E_vib_minus_ZPE"] = thermal
    if bad:
        alarm("P4-THM-007", nvib=nvib, T=T, **bad)


@guarded
def cbs_check(kind, zs, values, value, alpha=None):
    """P4-CBS-001..006 at the return of a driver_cbs_helper extrapolation (consecutive zeta, documented contract)."""
    if kind == "helgaker3":
        zlo, zmd, zhi = zs
        if not (zmd == zlo + 1 and zhi == zlo + 2):
            return
        res = pred_cbs_three_point(values[0], values[1], values[2], value)
        if res:
            alarm("P4-CBS-004", **res)
        return
    zlo, zhi = zs
    if zhi != zlo + 1 or alpha is None or not alpha > 0:
        return
    vlo, vhi = values
    if kind == "helgaker2":
        sid, ratio = "P4-CBS-001", math.exp(alpha * (zhi - zlo))
    elif kind == "truhlar2":
        sid, ratio = "P4-CBS-002", (zhi / zlo) ** alpha
    elif kind == "karton2":
        sid, ratio = "P4-CBS-003", math.exp(alpha * (math.sqrt(zhi) - math.sqrt(zlo)))
    elif kind == "corl2":
        sid, ratio = "P4-CBS-005", (zhi / zlo) ** alpha
    else:
        return
    res = pred_cbs_two_point(vlo, vhi, value, ratio)
    if res:
        alarm(sid, **res)
    res = pred_cbs_trend(vlo, vhi, value)
    if res:
        alarm("P4-CBS-006", kind=kind, **res)


@guarded
def scf_checks(wfn, e_scf):
    """P4-SCF-001..006 after SCF convergence (SO-basis matrices, per irrep)."""
    if _opt("SCF", "FRAC_START", 0):
        return
    if not _plain_run(wfn):
        return
    d = float(_opt("SCF", "D_CONVERGENCE", 1e-6))
    S = _blocks(wfn.S())
    Da, Db = _blocks(wfn.Da()), _blocks(wfn.Db())
    na, nb = wfn.nalpha(), wfn.nbeta()
    nm = max(1.0, na, nb)
    res = pred_spin_density_count(Da, S, na, 1e-8 * nm + d * nm)
    if res:
        alarm("P4-SCF-001", spin="alpha", **res)
    res = pred_spin_density_count(Db, S, nb, 1e-8 * nm + d * nm)
    if res:
        alarm("P4-SCF-001", spin="beta", **res)
    for D in (Da, Db):
        res = pred_idempotency(D, S, 1e-8 + 10.0 * d)
        if res:
            alarm("P4-SCF-002", **res)
            break
    Ca, Cb = _blocks(wfn.Ca()), _blocks(wfn.Cb())
    for C in (Ca, Cb):
        res = pred_orthonormality(C, S, 1e-8)
        if res:
            alarm("P4-SCF-003", **res)
            break
    if wfn.name() not in ("RHF", "UHF"):
        return
    Fa, Fb = _blocks(wfn.Fa()), _blocks(wfn.Fb())
    res = pred_fock_density_commutator(Fa, Da, S, 100.0 * d)
    if res is None:
        res = pred_fock_density_commutator(Fb, Db, S, 100.0 * d)
    if res:
        alarm("P4-SCF-004", **res)
    ea = [np.array(b, dtype=float) for b in wfn.epsilon_a().nph]
    eb = [np.array(b, dtype=float) for b in wfn.epsilon_b().nph]
    res = pred_eigen_equation(Fa, Ca, S, ea, 100.0 * d)
    if res is None:
        res = pred_eigen_equation(Fb, Cb, S, eb, 100.0 * d)
    if res:
        alarm("P4-SCF-006", **res)
    if (not _is_dft(wfn)) and wfn.functional().name() == "HF" and not wfn.has_variable("-D ENERGY"):
        if not (wfn.has_variable("PCM POLARIZATION ENERGY") and wfn.variable("PCM POLARIZATION ENERGY") != 0.0):
            enuc = wfn.molecule().nuclear_repulsion_energy()
            res = pred_hf_energy(e_scf, enuc, _blocks(wfn.H()), Da, Db, Fa, Fb, 1e-6 + 100.0 * d)
            if res:
                alarm("P4-SCF-005", **res)


_CI_METHODS = ("cisd", "cisdt", "cisdtq", "fci", "detci")


@guarded
def energy_checks(name, wfn):
    """Post-procedure checks from energy(): P4-MP2-001/002, P4-CI-001, P4-SAPT-001, P4-DFT-003."""
    from psi4 import core
    lower = str(name).lower()

    def var(k):
        return core.variable(k) if core.has_variable(k) else None

    if lower in ("mp2", "scs-mp2", "sos-mp2", "dfmp2", "conv-mp2", "df-mp2", "omp2", "mp2.5", "mp3") or "mp2" in lower:
        if str(_opt("SCF", "REFERENCE", "RHF")).upper() in ("RHF", "UHF") and wfn is not None:
            gapped = True
            try:
                for eps, nocc in ((wfn.epsilon_a(), wfn.nalphapi()), (wfn.epsilon_b(), wfn.nbetapi())):
                    occ, vir = [], []
                    for h in range(eps.nirrep()):
                        e = np.array(eps.nph[h], dtype=float)
                        occ += list(e[:nocc[h]])
                        vir += list(e[nocc[h]:])
                    if occ and vir and max(occ) >= min(vir):
                        gapped = False
            except Exception:  # noqa: BLE001
                gapped = False
            if gapped:
                for k, sid in (("MP2 CORRELATION ENERGY", "P4-MP2-001"),):
                    v = var(k)
                    if v is not None:
                        res = pred_nonpositive(float(v), 1e-10, k)
                        if res:
                            alarm(sid, **res)
                for k in ("MP2 SAME-SPIN CORRELATION ENERGY", "MP2 OPPOSITE-SPIN CORRELATION ENERGY"):
                    v = var(k)
                    if v is not None:
                        res = pred_nonpositive(float(v), 1e-10, k)
                        if res:
                            alarm("P4-MP2-002", **res)
    if lower in _CI_METHODS or (lower.startswith("ci") and lower[2:].isdigit()):
        e_scf, e_ci = var("SCF TOTAL ENERGY"), var("CURRENT ENERGY")
        if e_scf is not None and e_ci is not None:
            res = pred_ci_bound(float(e_ci), float(e_scf))
            if res:
                alarm("P4-CI-001", method=lower, **res)
    if lower.startswith(("sapt", "ssapt")):
        for k in ("SAPT DISP20 ENERGY", "SAPT IND20,R ENERGY"):
            v = var(k)
            if v is not None:
                res = pred_nonpositive(float(v), 1e-10, k)
                if res:
                    alarm("P4-SAPT-001", **res)
    if wfn is not None and _is_dft(wfn) and not _opt("SCF", "PERTURB_H"):
        q = wfn.V_potential().quadrature_values()
        # restricted: RHO_A is already the total density; unrestricted: alpha and beta are separate
        n_grid = q["RHO_A"] if wfn.same_a_b_dens() else q["RHO_A"] + q["RHO_B"]
        n = wfn.nalpha() + wfn.nbeta()
        if abs(n_grid - n) > 1e-3 * n:
            alarm("P4-DFT-003", n_grid=n_grid, n_electrons=n)


@guarded
def molecule_checks(mol):
    """P4-MOL-001/002 in vibanal_wfn (molecule is a core.Molecule or qcdb.Molecule)."""
    from psi4 import core
    nat = mol.natom()
    if nat < 2:
        return
    Z = np.array([mol.Z(i) for i in range(nat)])
    if np.any(Z <= 0):
        return  # ghost atoms
    m = np.array([mol.mass(i) for i in range(nat)])
    geom = np.asarray(mol.geometry())
    if not core.has_option_changed("THERMO", "ROTATIONAL_SYMMETRY_NUMBER"):
        ind = independent_rotational_symmetry_number(Z, m, geom)
        if ind is not None:
            res = pred_rot_symmetry_number(int(mol.rotational_symmetry_number()), int(ind))
            if res:
                alarm("P4-MOL-001", **res)
    res = pred_rot_constants(np.asarray(mol.rotational_constants(), dtype=float), independent_rotational_constants(m, geom))
    if res:
        alarm("P4-MOL-002", **res)


@guarded
def lineshape_check(lineshape_cls, domain_unused, gamma_fn, x0s):
    """P4-SPC-001: a line profile integrates to 1 on a private sinh-spaced grid."""
    for x0 in x0s:
        g = float(gamma_fn(x0))
        if not g > 0:
            continue
        t = np.linspace(-12.0, 12.0, 400001)
        x = x0 + g * np.sinh(t)
        prof = lineshape_cls(x, gamma_fn)
        y = prof.lineshape(x0)
        w = g * np.cosh(t)
        area = float(np.trapezoid(y * w, t)) if hasattr(np, "trapezoid") else float(np.trapz(y * w, t))
        if abs(area - 1.0) > 1e-4:
            alarm("P4-SPC-001", shape=lineshape_cls.__name__, x0=float(x0), gamma=g, area=area)
            return


@guarded
def dispersion_energy_check(disp, energy):
    """P4-DISP-001: two-body dispersion energy is non-positive."""
    if disp.fctldash in ("hf3c", "pbeh3c", "r2scan3c", "b973c") or "3c" in str(disp.dashlevel):
        return
    if not str(disp.dashlevel).startswith(("d2", "d3", "chg", "das2009", "das2010")):
        return
    if float(disp.dashparams.get("s9", 0.0)) != 0.0:
        return
    res = pred_nonpositive(float(energy), 1e-12, "dispersion_energy")
    if res:
        alarm("P4-DISP-001", dashlevel=disp.dashlevel, **res)


@guarded
def dispersion_gradient_check(disp, molecule, grad):
    """P4-DISP-002: dispersion gradient has zero net force and torque."""
    if disp.fctldash in ("hf3c", "pbeh3c", "r2scan3c", "b973c") or "3c" in str(disp.dashlevel):
        return
    g = np.asarray(grad)
    if g.ndim != 2 or g.shape[1] != 3 or not np.all(np.isfinite(g)):
        return
    r = np.asarray(molecule.geometry())
    scale = max(1.0, float(np.max(np.abs(g))))
    res = pred_net_force(g, 1e-9 * scale)
    if res is None:
        res = pred_net_torque(g, r, 1e-9 * scale)
    if res:
        alarm("P4-DISP-002", dashlevel=disp.dashlevel, **res)


@guarded
def functional_checks(func_dictionary):
    """P4-DFT-001/002: UEG limits of a functional defined by separate exchange and correlation components.

    Fused LibXC XC functionals and weights taken from LibXC (`use_libxc`) are outside the precondition.
    """
    x_funcs = func_dictionary.get("x_functionals", {})
    c_funcs = func_dictionary.get("c_functionals", {})
    x_hf = func_dictionary.get("x_hf", {})
    range_separated = any(p.get("omega", 0.0) != 0.0 for p in x_funcs.values()) \
        or x_hf.get("omega", 0.0) != 0.0 or "beta" in x_hf
    uses_libxc = "use_libxc" in x_hf or any("use_libxc" in p for p in x_funcs.values())
    if x_funcs and not range_separated and not uses_libxc:
        parts = [float(p.get("alpha", 1.0)) for p in x_funcs.values()]
        res = pred_weight_sum(parts + [float(x_hf.get("alpha", 0.0))], label="exchange_weight_sum")
        if res:
            alarm("P4-DFT-001", functional=func_dictionary.get("name"), **res)
    c_mp2 = func_dictionary.get("c_mp2", {})
    if c_funcs and not any(float(c_mp2.get(k, 0.0)) != 0.0 for k in ("ss", "os")):
        parts = [float(p.get("alpha", 1.0)) for p in c_funcs.values()]
        res = pred_weight_sum(parts + [float(c_mp2.get("alpha", 0.0))], label="correlation_weight_sum")
        if res:
            alarm("P4-DFT-002", functional=func_dictionary.get("name"), **res)
