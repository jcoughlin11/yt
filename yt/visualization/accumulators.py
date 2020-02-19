import numpy as np

from yt.units.yt_array import YTArray


def _accumulate_vector_field(path, field_vals):
    r"""
    This function integrates the given vector field along the given
    path p. The integral is done in a piecewise manner
    (segment-by-segment) so as to be able to store the accumulated
    values from each of the previous segments.

    For a vector field, the line integral is:

    ..math::

        I = \int_C \vec{a} \cdot d\vec{r}

    where :math:`C` is the path being integrated along, :math:`\vec{a}` is
    the vector field being integrated, and :math:`d\vec{r}` points along the
    path :math:`C`. This is equivalent to:
    
    ..math::

        \lim_{N \rightarrow \infty} \sum_{p=1}^N \vec{a}(x_p, y_p, ...) \cdot \Delta \vec{r}_p
        
    with the understanding that :math:`\Delta \vec{r}_p \rightarrow 0` as
    :math:`N \rightarrow \infty`.
        
    The vector pointing along the segment connecting two adjacent points is:
    
    ..math::
    
        \Delta\vec{r}_p = (\vec{r}_{p+1} - \vec{r}_p)
        
    so the full dot product can be written out as:

    ..math::
        I = \lim_{N \rightarrow \infty}\sum_{p=1}^N\sum_{i=1}^n a_{p,i}(r_{p+1,i}-r_{p,i})
        
    where :math:`n` is the number of dimensions.

    This can be done as a matrix operation. If :math:`\vec{r} \equiv \vec{r}_{p+1} - \vec{r}_p`
    then the above sum is (leaving out the limit):
    
    ..math::
    
        I = \vec{a}_1 \dot \vec{r}_1 + \vec{a}_2 \cdot \vec{r}_2 \ldots \vec{a}_N \cdot \vec{r}_N
        
    If we write the matrix :math:`A = ([a1x, a1y, ...], [a2x, a2y, ...], ...)`
    and :math:`R = ([r1x, r1y, ...], [r2x, r2y, ...], ...)^T`, then the dot
    products in the sum are the diagonal elements of the resulting matrix
    multiplication :math:`AR`. The accumulation is then obtained by doing a
    cumsum of this diagonal.

    Parameters
    ----------
    p : YTArray 
        The path to be integrated along

    field_vals : YTArray 
        An array containing the components of the vector field to be
        integrated. The values are sampled at the starting point of
        each path segment as well as the endpoint for the last segment

    Returns
    -------
        accum : YTArray 
            The cumulative value of the field integral at each path
            segment
    """
    accum = np.cumsum(np.diag(np.dot(field_vals[:-1], (p[1:] - p[:-1]).T)))
    return accum


def _accumulate_scalar_field(p, field_vals):
    r"""
    This function integrates a scalar field along a path. It uses a
    similar method to that in _accumulate_vector_field, but the integral
    is now:
    
    ..math::
    
        I = \int_C \phi(x1,x2,...,xn)d\vec{r}

    Parameters
    ----------
    p : YTArray 
        The path to be integrated along

    fieldVals : YTArray 
        An array containing the values of the scalar field to be
        integrated at the location of the starting point of each
        path segment as well as the endpoint for the last segment

    Returns
    -------
    accum : YTArray 
        The cumulative value of the field integral at each path
        segment 
    """
    accum = field_vals[:-1] * (p[1:] - p[:-1])
    # Since this integral results in a vector, I think the accumulation
    # should add the vectors together. The default for np.cumsum is
    # axis=None, which performs the cumulative summation along the
    # flattened array. Axis=0 adds the columns, which is akin to
    # performing a cumulative summation of each of the components
    accum = np.cumsum(accum, axis=0)
    return accum


class Accumulators:
    r"""
    Container for creating and storing the path integrals of various
    fields along various paths.

    The class takes in a list of user-defined paths and a dataset. From
    these, the user can compute the integral of any field in the dataset
    along the given paths. The results of these path integrals are
    stored in a cumulative fashion, which means that, once computed, the
    accumulated value of the field is stored at each point along the
    path, allowing the user to query this information.

    Attributes
    ----------
    pass

    Methods
    -------
    pass

    Examples
    --------
    pass
    """
    def __init__(self, paths, ds):
        self.paths      = paths
        self.ds         = ds
        self.ad         = ds.all_data()
        self.accum      = []

    def _join_field_components(self, field):
        r"""
        This function takes the disparate components of the field,
        which are stored in the three different arrays fx, fy, and fz,
        and joins them into one ndarray. The result needs to have a
        shape of  (nGridCells, nDims)

        Parameters
        ----------
        field : iterable
            Contains the components of the field to be accumulated
            along the desired paths.

        Returns
        -------
        field_grid_vals : np.ndarray
            An array of shape (n_cells, n_field_components). This form
            allows for easier matrix manipulation.
        """
        # Calling np.stack results in the unit-ful array having
        # units of 'dimensionless.' It's easier to to just save the
        # original units, work with the raw values, and then add the
        # units back in at the end
        unit = self.ad[field[0]].units
        # Strip units from coordinates
        x_field = self.ad[field[0]].d
        y_field = self.ad[field[1]].d
        z_field = self.ad[field[2]].d
        field_grid_vals = np.stack((x_field, y_field, z_field), axis=1)
        # Add units back
        field_grid_vals = YTArray(field_grid_vals, unit)
        return field_grid_vals

    def accumulate(self, field, is_vector=None):
        r"""
        This function is the driver function for integrating the desired
        field along each path in the bundle.

        Parameters
        ----------
        field : YTFieldData, tuple
            The name of the field to integrate along the paths. If this
            is a vector field, field should be an iterable containing
            the components of the field (i.e, field[0] is the x-component
            data, field[1] is the y-component, etc.).

        is_vector : bool
            If True, field is a vector field. If False, then field is
            a scalar field.

        Raises
        ------
        ValueError
            If is_vector is not set.
        """
        if is_vector is None:
            raise ValueError("`is_vector` parameter not set.")
        if is_vector:
            # Join the components together into a more convenient form
            field = self._join_field_components(field)
        # Loop over each path
        for p in self.paths:
            # Calculate the values of the field at each point
            field_vals = ds.find_field_values_at_points(field, p)
            # Integrate the field along the path based on whether or not
            # the field is a vector or scalar field
            if is_vector:
                self.accum.append(_accumulate_vector_field(p, field_vals))
            else:
                self.accum.append(_accumulate_scalar_field(p, field_vals))
