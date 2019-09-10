"""
title: test_athena_pp.py
Purpose: Athena++ frontend tests
Notes:
    Copyright (c) 2013, yt Development Team.
    Distributed under the terms of the Modified BSD License.
    The full license is in the file COPYING.txt, distributed with this
    software.
"""
from collections import OrderedDict

import numpy as np
import pytest

from yt.convenience import load
from yt.frontends.athena_pp.api import AthenaPPDataset
from yt.testing import \
    assert_allclose, \
    requires_file
    units_override_check
import yt.utilities.answer_testing.framework as fw
from yt.utilities.answer_testing import utils


# Test data
disk = "KeplerianDisk/disk.out1.00000.athdf"
AM06 = "AM06/AM06.out1.00400.athdf"


# Answer file
answer_file = 'athena_pp_answers.yaml'


#============================================
#                TestAthenaPP
#============================================
@pytest.mark.skipif(not pytest.config.getvalue('--with-answer-testing'),
    reason="--with-answer-testing not set.")
class TestAthenaPP(fw.AnswerTest):
    """
    Container for athena++ frontent tests.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # test_disk
    #-----
    @utils.requires_ds(disk)
    def test_disk(self, ds_disk):
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
        hashes = OrderedDict()
        hashes['generic_array'] = OrderedDict()
        fields = ("density", "velocity_r")
        dd = ds_disk.all_data()
        vol = (ds_disk.domain_right_edge[0]**3-ds_disk.domain_left_edge[0]**3)/3.0
        vol *= np.cos(ds_disk.domain_left_edge[1])-np.cos(ds_disk.domain_right_edge[1])
        vol *= ds_disk.domain_right_edge[2].v-ds_disk.domain_left_edge[2].v
        assert_allclose(dd.quantities.total_quantity("cell_volume"), vol)
        for field in fields:
            def field_func(name):
                return dd[field]
            ga_hd = utils.generate_hash(
                self.generic_array_test(ds_disk, field_func, args=[field])
            )
            hashes['generic_array'][field] = ga_hd
        hashes = {'disk' : hashes}
        utils.handle_hashes(self.save_dir, answer_file, hashes, self.answer_store)

    #-----
    # test_AM06
    #-----
    @utils.requires_ds(AM06)
    def test_AM06(self, ds_AM06):
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
        # Arrays for testing
        axes = [0, 1, 2]
        center = "max"
        ds_objs = [None, ("sphere", (center, (0.1, 'unitary')))]
        weights = [None, "density"]
        fields = ("temperature",
            "density",
            "velocity_magnitude",
            "magnetic_field_x"
        )
        # Run the small_patch_amr test suite
        hashes = self.small_patch_amr(ds_AM06, fields, weights, axes, ds_objs)
        # Save or compare answer
        utils.handle_hashes(self.save_dir, answer_file, hashes, self.answer_store)

    #-----
    # test_AM06_override
    #-----
    @requires_file(AMO06)
    def test_AM06_override(self):
        """
        Verify that overriding units causes derived unit values to be
        updated. See issue #1259.

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
        uo_AM06 = {
            'length_unit': (1.0, 'kpc'),
            'mass_unit': (1.0, 'Msun'),
            'time_unit': (1.0, 'Myr'),
        }
        ds = load(AM06, units_override=uo_AM06)
        assert float(ds.magnetic_unit.in_units('gauss')) == 9.01735778342523e-08

    #-----
    # test_units_override
    #-----
    @requires_file(AM06)
    def test_units_override(self, ds_AM06):
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
        units_override_check(ds_AM06, AM06)

    #-----
    # test_AthenaPPDataset
    #-----
    @requires_file(AM06)
    def test_AthenaPPDataset(self, ds_AM06):
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
        assert isinstance(ds_AM06, AthenaPPDataset)
