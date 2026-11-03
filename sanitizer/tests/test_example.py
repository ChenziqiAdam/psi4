import psi4


def test_water_gradient():
    psi4.core.be_quiet()
    psi4.geometry("0 1\nO\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1")
    psi4.set_options({"basis": "6-31g", "scf_type": "pk"})
    g = psi4.gradient("scf")
    assert g.np.shape == (3, 3)
