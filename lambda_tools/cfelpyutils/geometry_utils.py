#    This file is part of cfelpyutils.
#
#    cfelpyutils is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    cfelpyutils is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with cfelpyutils.  If not, see <http://www.gnu.org/licenses/>.
"""
Geometry utilities.

Functions that load, manipulate and apply geometry information to
detector pixel data.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections

import numpy


def compute_pixel_maps(geometry):
    """
    Compute pixel maps from a CrystFEL geometry object.

    Take as input a CrystFEL-style geometry object (A dictionary
    returned by the function load_crystfel_geometry function in the
    crystfel_utils module) and return a PixelMap tuple . The origin the
    reference system used by the pixel maps is set at the beam
    interaction point.

    Args:

        geometry (dict): A CrystFEL geometry object (A dictionary
            returned by the
            :obj:`cfelpyutils.crystfel_utils.load_crystfel_geometry`
            function).

    Returns:

        Tuple[ndarray, ndarray, ndarray] A tuple containing the pixel
        maps. The first two fields, named "x" and "y" respectively,
        store the pixel maps for the x coordinate and the y coordinate.
        The third field, named "r", is instead a pixel map storing the
        distance of each pixel in the data array from the center of the
        reference system.
    """
    # Determine the max fs and ss in the geometry object.
    max_slab_fs = numpy.array([
        geometry['panels'][k]['max_fs']
        for k in geometry['panels']
    ]).max()

    max_slab_ss = numpy.array([
        geometry['panels'][k]['max_ss']
        for k in geometry['panels']
    ]).max()

    # Create empty arrays, of the same size of the input data, that
    # will store the x and y pixel maps.
    x_map = numpy.zeros(
        shape=(max_slab_ss + 1, max_slab_fs + 1),
        dtype=numpy.float32  # pylint: disable=E1101
    )

    y_map = numpy.zeros(
        shape=(max_slab_ss + 1, max_slab_fs + 1),
        dtype=numpy.float32  # pylint: disable=E1101
    )

    # Iterate over the panels. For each panel, determine the pixel
    # indeces, then compute the x,y vectors using a comples notation.
    for pan in geometry['panels']:
        i, j = numpy.meshgrid(
            numpy.arange(
                geometry['panels'][pan]['max_ss'] -
                geometry['panels'][pan]['min_ss'] +
                1
            ),
            numpy.arange(
                geometry['panels'][pan]['max_fs'] -
                geometry['panels'][pan]['min_fs'] +
                1
            ),
            indexing='ij'
        )

        d_x = (geometry['panels'][pan]['fsy'] +
               1J * geometry['panels'][pan]['fsx'])

        d_y = (geometry['panels'][pan]['ssy'] +
               1J * geometry['panels'][pan]['ssx'])

        r_0 = (geometry['panels'][pan]['cny'] +
               1J * geometry['panels'][pan]['cnx'])
        cmplx = i * d_y + j * d_x + r_0

        x_map[
            geometry['panels'][pan]['min_ss']:
            geometry['panels'][pan]['max_ss'] + 1,
            geometry['panels'][pan]['min_fs']:
            geometry['panels'][pan]['max_fs'] + 1
        ] = cmplx.imag

        y_map[
            geometry['panels'][pan]['min_ss']:
            geometry['panels'][pan]['max_ss'] + 1,
            geometry['panels'][pan]['min_fs']:
            geometry['panels'][pan]['max_fs'] + 1
        ] = cmplx.real

    # Finally, compute the values for the radius pixel map.
    r_map = numpy.sqrt(numpy.square(x_map) + numpy.square(y_map))
    PixelMaps = collections.namedtuple(
        typename='PixelMaps',
        field_names=['x', 'y', 'r']
    )
    return PixelMaps(x_map, y_map, r_map)


def apply_pixel_maps(data, pixel_maps, output_array=None):
    """
    Apply geometry in pixel map format to the input data.

    Turn an array of detector pixel values into an array
    containing a representation of the physical layout of the detector.

    Args:

        data (ndarray): array containing the data on which the geometry
            will be applied.

        pixel_maps (PixelMaps): a pixelmap tuple, as returned by the
            :obj:`compute_pixel_maps` function in this module.

        output_array (Optional[ndarray]): a preallocated array (of
            dtype numpy.float32) to store the function output. If
            provided, this array will be filled by the function and
            and returned to the user. If not provided, the function
            will create a new array automatically and return it to the
            user. Defaults to None (No array provided).

    Returns:

        ndarray: a numpy.float32 array containing the geometry
        information applied to the input data (i.e.: a representation
        of the physical layout of the detector).
    """
    # If no output array was provided, create one.
    if output_array is None:
        output_array = numpy.zeros(
            shape=data.shape,
            dtype=numpy.float32  # pylint: disable=E1101
        )

    # Apply the pixel map geometry information the data, then return
    # the resulting array.
    output_array[pixel_maps.y, pixel_maps.x] = data
    return output_array


def compute_minimum_array_size(pixel_maps):
    """
    Compute the minimum size of an array that can store the applied
    geometry.

    Return the minimum size of an array that can store data on which
    the geometry information described by the pixel maps has been
    applied.

    The returned array shape is big enough to display all the input
    pixel values in the reference system of the physical detector. The
    array is supposed to be centered at the center of the reference
    system of the detector (i.e: the beam interaction point).

    Args:

        Tuple[ndarray, ndarray, ndarray]: a named tuple containing the
            pixel maps. The first two fields, "x" and "y", should store
            the pixel maps for the x coordinateand the y coordinate.
            The third, "r", should instead store the distance of each
            pixel in the data array from the center of the reference
            system.

    Returns:

        Tuple[int, int]: a numpy-style shape tuple storing the minimum
        array size.
    """
    # Find the largest absolute values of x and y in the maps. Since
    # the returned array is centered on the origin, the minimum array
    # size along a certain axis must be at least twice the maximum
    # value for that axis. 2 pixels are added for good measure.
    x_map, y_map = pixel_maps.x, pixel_maps.y
    y_minimum = 2 * int(max(abs(y_map.max()), abs(y_map.min()))) + 2
    x_minimum = 2 * int(max(abs(x_map.max()), abs(x_map.min()))) + 2

    # Return a numpy-style tuple with the computed shape.
    return (y_minimum, x_minimum)


def adjust_pixel_maps_for_pyqtgraph(pixel_maps):
    """
    Adjust pixel maps for visualization of the data in a pyqtgraph
    widget.

    The adjusted maps can be used for a Pyqtgraph ImageView widget.

    Args:

        Tuple[ndarray, ndarray, ndarray]: a named tuple containing the
        pixel maps. The first two fields, "x" and "y", should store the
        pixel maps for the x coordinateand the y coordinate. The third,
        "r", should instead store the distance of each pixel in the
        data array from the center of the reference system.

    Returns:

        Tuple[ndarray, ndarray] A tuple containing the pixel
            maps. The first two fields, named "x" and "y" respectively,
            store the pixel maps for the x coordinate and the y
            coordinate. The third field, named "r", is instead a pixel
            map storing the distance of each pixel in the data array
            from the center of the reference system.
    """
    # Essentially, the origin of the reference system needs to be
    # moved from the beam position to the top-left of the image that
    # will be displayed. First, compute the size of the array used to
    # display the data, then use this information to estimate the
    # magnitude of the shift that needs to be applied to the origin of
    # the system.
    min_shape = compute_minimum_array_size(pixel_maps)
    new_x_map = numpy.array(
        object=pixel_maps.x,
        dtype=numpy.int
    ) + min_shape[1] // 2 - 1

    new_y_map = numpy.array(
        object=pixel_maps.y,
        dtype=numpy.int
    ) + min_shape[0] // 2 - 1

    PixelMapsForIV = collections.namedtuple(
        typename='PixelMapsForIV',
        field_names=['x', 'y']
    )
    return PixelMapsForIV(new_x_map, new_y_map)
