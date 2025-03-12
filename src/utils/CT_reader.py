import vtk
from vtk.util import numpy_support


def read_ct_volume(CT_folder):
    # Create DICOM reader
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(CT_folder)
    reader.Update()

    vtk_image = reader.GetOutput()
    dims = vtk_image.GetDimensions()
    spacing = vtk_image.GetSpacing()
    origin = vtk_image.GetOrigin()
    extent = vtk_image.GetExtent()

    print("CT Volume Info:")
    print(f"Dimensions: {dims}")
    print(f"Spacing: {spacing}")
    print(f"Origin: {origin}")
    print(f"Extent: {extent}")

    # Get the image orientation from DICOM
    reader_info = reader.GetImagePositionPatient()
    image_position = reader_info if reader_info else origin
    print(f"Image Position (Patient): {image_position}")

    # Store CT data and metadata
    return {
        'image': vtk_image,
        'dims': dims,
        'spacing': spacing,
        'origin': origin,
        'position': image_position,
        'extent': extent
    }