"""Independent spectral/sequence anchors for previously failing DMD inputs."""
import numpy as np
import pytest
from modal_current.dmd import compute_dmd


def test_negative_real_eigenvalue_is_a_finite_nyquist_mode():
    dt=.125
    data=(-.8)**np.arange(16)[None,:]
    with np.errstate(all="raise"):
        result=compute_dmd(data,dt=dt,rank=1)
        assert result.growth_rates[0] == pytest.approx(np.log(.8)/dt)
        assert abs(result.frequencies[0]) == pytest.approx(np.pi/dt)
        assert np.all(np.isfinite(result.continuous_eigenvalues))
        np.testing.assert_allclose(result.reconstruct_training(),data,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(result.reconstruct(dt*np.arange(16)),data,rtol=1e-12,atol=1e-14)


def test_zero_mode_preserves_initial_impulse_then_vanishes():
    data=np.array([[1.,0.,0.,0.,0.]])
    with np.errstate(all="raise"):
        result=compute_dmd(data,dt=.1)
        np.testing.assert_array_equal(result.eigenvalues,[0.])
        np.testing.assert_allclose(result.reconstruct_training(),data,rtol=0,atol=1e-14)
    for times in ([0.,.1],[-.1,0.],[.05]):
        with pytest.raises(ValueError,match="zero eigenvalues"):
            result.reconstruct(times)
    with pytest.raises(ValueError,match="zero eigenvalues"):
        _=result.continuous_eigenvalues


def test_mixed_zero_and_decaying_modes_have_finite_discrete_reconstruction():
    data=np.vstack([np.r_[1.,np.zeros(11)],.6**np.arange(12)])
    result=compute_dmd(data,dt=.2,rank=2)
    # Exact zero is guaranteed only for the scalar impulse anchor above;
    # fitted mixed spectra can contain a tiny rounded nonzero eigenvalue.
    np.testing.assert_allclose(result.reconstruct_training(),data,rtol=1e-12,atol=1e-13)
    assert np.all(np.isfinite(result.reconstruct_training()))


@pytest.mark.parametrize("data,rank", [
    (np.array([[1.,2.,3.,4.],[0.,0.,0.,0.]]),2),
    (np.array([[1.,2.,3.,4.],[2.,4.,6.,8.]]),2),
    (np.zeros((2,4)),1),
])
def test_explicit_rank_above_numerical_rank_is_rejected_before_inversion(data,rank):
    with np.errstate(all="raise"), pytest.raises(ValueError,match="exceeds numerical rank"):
        compute_dmd(data,dt=.1,rank=rank)
