"""
Title:   test_enzo.py
Purpose: Contains Enzo frontend tests
Notes:
    Copyright (c) 2013, yt Development Team.
    Distributed under the terms of the Modified BSD License.
    The full license is in the file COPYING.txt, distributed with this
    software.
"""
from collections import OrderedDict

import pytest

from yt.frontends.enzo.api import EnzoDataset
from yt.frontends.enzo.fields import NODAL_FLAGS
from yt.testing import \
    assert_allclose_units, \
    assert_almost_equal, \
    assert_equal, \
    requires_file, \
    units_override_check
from yt.visualization.plot_window import SlicePlot
import yt.utilities.answer_testing.framework as fw
from yt.utilities.answer_testing import utils


# Files containing data to be used in tests. Paths are relative to
# yt test_data_dir
toro1d = "ToroShockTube/DD0001/data0001"
kh2d = "EnzoKelvinHelmholtz/DD0011/DD0011"
m7 = "DD0010/moving7_0010"
g30 = "IsolatedGalaxy/galaxy0030/galaxy0030"
enzotiny = "enzo_tiny_cosmology/DD0046/DD0046"
ecp = "enzo_cosmology_plus/DD0046/DD0046"
two_sphere_test = 'ActiveParticleTwoSphere/DD0011/DD0011'
active_particle_cosmology = 'ActiveParticleCosmology/DD0046/DD0046'
mhdctot = "MHDCTOrszagTang/DD0004/data0004"
dnz = "DeeplyNestedZoom/DD0025/data0025"
p3mini = "PopIII_mini/DD0034/DD0034"


#============================================
#                 TestEnzo
#============================================
@pytest.mark.skipif(not pytest.config.getvalue('--with-answer-testing'),
    reason="--with-answer-testing not set.")
@pytest.mark.usefixtures('answer_file')
class TestEnzo(fw.AnswerTest):
    #-----
    # test_toro1d
    #-----
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(toro1d)
    def test_toro1d(self, a, d, w, f, ds_toro1d):
        # Run the small_patch_amr test suite
        self.hashes.update(self.small_patch_amr(ds_toro1d, f, w, a, d))

    #-----
    # test_kh2d
    #-----
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(kh2d)
    def test_kh2d(self, a, d, w, f, ds_kh2d):
        # Run the small_patch_amr test suite
        self.hashes.update(self.small_patch_amr(ds_kh2d, f, w, a, d))

    #-----
    # test_moving7
    #-----
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(m7)
    def test_moving7(self, a, d, w, f, ds_m7):
        # Run the small_patch_amr test suite
        self.hashes.update(self.small_patch_amr(ds_m7, f, w, a, d))

    #-----
    # test_galaxy0030
    #-----
    @pytest.mark.skipif(not pytest.config.getvalue('--answer-big-data'),
        reason="Skipping because --answer-big-data was not set."
    )
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(g30)
    def test_galaxy0030(self, a, d, w, f, ds_g30):
        # Color conservation test
        self.color_conservation_test(ds_g30)
        # Run the big patch amr test suite
        self.hashes.update(self.big_patch_amr(ds_g30, f, w, a, d))

    #-----
    # test_simulated_halo_mass_function
    #-----
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(enzotiny)
    def test_simulated_halo_mass_function(self, finder, ds_enzotiny):
        shmf_hd = self.simulated_halo_mass_function_test(ds_enzotiny, finder)
        self.hashes.update({'simulated_halo_mass_function' : shmf_hd})

    #-----
    # test_analytic_halo_mass_function
    #-----
    @pytest.mark.usefixtures('hashing')
    @utils.requires_ds(enzotiny)
    def test_analytic_halo_mass_function(self, fit, ds_enzotiny):
        ahmf_hd = self.analytic_halo_mass_function_test(ds_enzotiny, fit)
        self.hashes.update({'analytic_halo_mass_function' : ahmf_hd})

    #-----
    # test_ecp
    #-----
    @pytest.mark.skipif(not pytest.config.getvalue('--answer-big-data'),
        reason="Skipping test_jet because --answer-big-data was not set."
    )
    @utils.requires_ds(ecp)
    def test_ecp(self, ds_ecp):
        self.color_conservation_test(ds_ecp)

    #-----
    # test_units_override
    #-----
    @requires_file(enzotiny)
    def test_units_override(self, ds_enzotiny):
        units_override_check(ds_enzotiny, enzotiny)

    #-----
    # test_nuclei_density_fields
    #-----
    @pytest.mark.skipif(not pytest.config.getvalue('--answer-big-data'),
        reason="Skipping test_jet because --answer-big-data was not set."
    )
    @utils.requires_ds(ecp)
    def test_nuclei_density_fields(self, ds_ecp):
        ad = ds_ecp.all_data()
        # Hydrogen
        hd1 = utils.generate_hash(ad["H_nuclei_density"].tostring())
        hd2 = utils.generate_hash((ad["H_number_density"] +
            ad["H_p1_number_density"]).tostring())
        assert hd1 == hd2
        hd1 = utils.generate_hash(ad["He_nuclei_density"].tostring())
        hd2 = utils.generate_hash((ad["He_number_density"] +
            ad["He_p1_number_density"] +
            ad["He_p2_number_density"]).tostring()
        )
        assert hd1 == hd2

    #-----
    # test_EnzoDataset
    #-----
    @requires_file(enzotiny)
    def test_EnzoDataset(self, ds_enzotiny):
        assert isinstance(ds_enzotiny, EnzoDataset)

    #-----
    # test_active_particle_dataset
    #-----
    @requires_file(two_sphere_test)
    @requires_file(active_particle_cosmology)
    def test_active_particle_datasets(self, ds_two_sphere_test, ds_active_particle_cosmology):
        # Set up lists for comparison
        pfields = ['GridID', 'creation_time', 'dynamical_time',
                   'identifier', 'level', 'metallicity', 'particle_mass']
        pfields += ['particle_position_%s' % d for d in 'xyz']
        pfields += ['particle_velocity_%s' % d for d in 'xyz']
        acc_part_fields = \
            [('AccretingParticle', pf) for pf in ['AccretionRate'] + pfields]
        real_acc_part_fields = sorted(
            [f for f in ds_two_sphere_test.field_list if f[0] == 'AccretingParticle'])
        # Set up lists for comparison
        apcos_fields = [('CenOstriker', pf) for pf in pfields]
        real_apcos_fields = sorted(
            [f for f in ds_active_particle_cosmology.field_list if f[0] == 'CenOstriker'])
        apcos_pcounts = {'CenOstriker': 899755, 'DarkMatter': 32768}
        assert 'AccretingParticle' in ds_two_sphere_test.particle_types_raw
        assert 'io' not in ds_two_sphere_test.particle_types_raw
        assert 'all' in ds_two_sphere_test.particle_types
        assert_equal(len(ds_two_sphere_test.particle_unions), 1)
        assert_equal(acc_part_fields, real_acc_part_fields)
        assert_equal(['CenOstriker', 'DarkMatter'], ds_active_particle_cosmology.particle_types_raw)
        assert 'all' in ds_active_particle_cosmology.particle_unions
        assert_equal(apcos_fields, real_apcos_fields)
        assert_equal(ds_active_particle_cosmology.particle_type_counts, apcos_pcounts)

    #-----
    # test_face_centered_mhdct_fields
    #-----
    @requires_file(mhdctot)
    def test_face_centered_mhdct_fields(self, ds_mhdctot):
        ad = ds_mhdctot.all_data()
        grid = ds_mhdctot.index.grids[0]
        dims = ds_mhdctot.domain_dimensions
        dims_prod = dims.prod()
        for field, flag in NODAL_FLAGS.items():
            assert_equal(ad[field].shape, (dims_prod, 2*sum(flag)))
            assert_equal(grid[field].shape, tuple(dims) + (2*sum(flag),))
        # Average of face-centered fields should be the same as
        # cell-centered field
        assert (ad['BxF'].sum(axis=-1)/2 == ad['Bx']).all()
        assert (ad['ByF'].sum(axis=-1)/2 == ad['By']).all()
        assert (ad['BzF'].sum(axis=-1)/2 == ad['Bz']).all()

    #-----
    # test_deeply_nested_zoom
    #-----
    @utils.requires_ds(dnz)
    def test_deeply_nested_zoom(self, ds_dnz):
        # Carefully chosen to just barely miss a grid in the middle of
        # the image
        center = [0.4915073260199302, 0.5052605316800006, 0.4905805557500548]
        plot = SlicePlot(ds_dnz, 'z', 'density', width=(0.001, 'pc'),
                         center=center)
        image = plot.frb['density']
        assert (image > 0).all()
        v, c = ds_dnz.find_max('density')
        assert_allclose_units(v, ds_dnz.quan(0.005878286377124154, 'g/cm**3'))
        c_actual = [0.49150732540021, 0.505260532936791, 0.49058055816398]
        c_actual = ds_dnz.arr(c_actual, 'code_length')
        assert_allclose_units(c, c_actual)
        assert_equal(max([g['density'].max() for g in ds_dnz.index.grids]), v)

    #-----
    # test_2d_grid_shape
    #-----
    @requires_file(kh2d)
    def test_2d_grid_shape(self, ds_kh2d):
        r"""See issue #1601: we want to make sure that accessing data on
        a grid object returns a 3D array with a dummy dimension
        """
        g = ds_kh2d.index.grids[1]
        assert g['density'].shape == (128, 100, 1)

    #-----
    # test_nonzero_omega_radiation
    #-----
    @requires_file(p3mini)
    def test_nonzero_omega_radiation(self, ds_p3mini):
        r"""Test support for non-zero omega_radiation cosmologies.
        """
        err_msg = "Simulation time not consistent with cosmology calculator."
        t_from_z = ds_p3mini.cosmology.t_from_z(ds_p3mini.current_redshift)
        tratio = ds_p3mini.current_time / t_from_z
        assert_equal(ds_p3mini.omega_radiation, ds_p3mini.cosmology.omega_radiation)
        assert_almost_equal(tratio, 1, 4, err_msg=err_msg)
