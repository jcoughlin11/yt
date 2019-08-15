"""
Title: test_flash.py
Purpose: FLASH frontend tests
Notes:
    Copyright (c) 2013, yt Development Team.
    Distributed under the terms of the Modified BSD License.
    The full license is in the file COPYING.txt, distributed with this
    software.
"""
from collections import OrderedDict

import numpy as np

from yt.frontends.flash.api import FLASHDataset, \
    FLASHParticleDataset
from yt.testing import \
    assert_equal, \
    requires_file, \
    units_override_check

import framework as fw
import utils

# Test data
sloshing = "GasSloshingLowRes/sloshing_low_res_hdf5_plt_cnt_0300"
wt = "WindTunnel/windtunnel_4lev_hdf5_plt_cnt_0030"
fid_1to3_b1 = "fiducial_1to3_b1/fiducial_1to3_b1_hdf5_part_0080"
dens_turb_mag = 'DensTurbMag/DensTurbMag_hdf5_plt_cnt_0015'


#============================================
#                 TestFlash
#============================================
class TestFlash(fw.AnswerTest):
    """
    Container for flash frontend answer tests.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # test_sloshing
    #-----
    @pytest.mark.skipif(not pytest.config.getvalue('--answer-big-data'),
        reason="Skipping test_jet because --answer-big-data was not set."
    )
    @utils.requires_ds(sloshing)
    def test_sloshing(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Set up arrays for testing
        axes = [0, 1, 2]
        center = "max"
        ds_objs = [None, ("sphere", (center, (0.1, 'unitary')))]
        weights = [None, "density"]
        fields = ("temperature", "density", "velocity_magnitude")
        ds = utils.data_dir_load(sloshing)
        assert_equal(str(ds) == "sloshing_low_res_hdf5_plt_cnt_0300")
        # Run the small_patch_amr test suite
        hashes = self.small_patch_amr(ds, fields, weights, axes, ds_objs)
        # Save or compare answer
        utils.handle_hashes(self.save_dir, 'sloshing', hashes, self.answer_store)

    #-----
    # test_wind_tunnel
    #-----
    @utils.requires_ds(wt)
    def test_wind_tunnel(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Set up arrays for testing
        axes = [0, 1, 2]
        center = "max"
        ds_objs = [None, ("sphere", (center, (0.1, 'unitary')))]
        weights = [None, "density"]
        fields = ("temperature", "density")
        ds = utils.data_dir_load(wt)
        assert_equal(str(ds) == "windtunnel_4lev_hdf5_plt_cnt_0030")
        # Run the small_patch_amr test suite
        hashes = self.small_patch_amr(ds, fields, weights, axes, ds_objs)
        # Save or compare answer
        utils.handle_hashes(self.save_dir, 'wind-tunnel', hashes, self.answer_store)

    #-----
    # test_FLASHDataset
    #-----
    @requires_file(wt)
    def test_FLASHDataset(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        assert isinstance(utils.data_dir_load(wt), FLASHDataset)

    #-----
    # test_units_override
    #-----
    @requires_file(sloshing)
    def test_units_override(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        units_override_check(sloshing)

    #-----
    # test_FLASHParticleDataset
    #-----
    @requires_file(fid_1to3_b1)
    def test_FLASHParticleDataset(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        assert isinstance(utils.data_dir_load(fid_1to3_b1),
            FLASHParticleDataset)

    #-----
    # test_Flash25_dataset
    #-----
    @requires_file(dens_turb_mag)
    def test_FLASH25_dataset(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        ds = utils.data_dir_load(dens_turb_mag)
        assert_equal(ds.parameters['time'], 751000000000.0)
        assert_equal(ds.domain_dimensions, np.array([8, 8, 8]))
        assert_equal(ds.domain_left_edge,
            ds.arr([-2e18, -2e18, -2e18], 'code_length'))
        assert_equal(ds.index.num_grids, 73)
        dd = ds.all_data()
        dd['density']

    #-----
    # test_fid_1to3_b1
    #-----
    @pytest.mark.skipif(not pytest.config.getvalue('--answer-big-data'),
        reason="Skipping test_jet because --answer-big-data was not set."
    )
    @utils.requires_ds(fid_1to3_b1)
    def test_fid_1to3_b1(self):
        """
        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        fid_1to3_b1_fields = OrderedDict(
            [
                (("deposit", "all_density"), None),
                (("deposit", "all_count"), None),
                (("deposit", "all_cic"), None),
                (("deposit", "all_cic_velocity_x"), ("deposit", "all_cic")),
                (("deposit", "all_cic_velocity_y"), ("deposit", "all_cic")),
                (("deposit", "all_cic_velocity_z"), ("deposit", "all_cic")),
            ]
        )
        ds = utils.data_dir_load(fid_1to3_b1)
        # Run the sph_answer test suite
        hashes = self.sph_answer(ds,
            'fiducial_1to3_b1_hdf5_part_0080',
            6684119,
            fid_1to3_b1_fields
        )
        # Save or compare answer
        utils.handle_hashes(self.save_dir, 'fid_1to3_b1', hashes, self.answer_store)
