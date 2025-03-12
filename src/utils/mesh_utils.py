import vtk
import numpy as np


def preprocess_mesh(mesh, dataset):
    if dataset == 'ATLAS':
        transform_scale = vtk.vtkTransform()
        # Scale the mesh; not sure why this is necessary
        transform_scale.Scale(.001, .001, .001)
        mesh.transform(transform_scale)
        # Unsure why this is necessary, but mesh seems to be flipped; mirror the mesh along the x axis
        transform_mirror = vtk.vtkTransform()
        transform_mirror.Scale(-1, 1, 1)
        mesh.transform(transform_mirror)

def undo_preprocess_mesh(mesh, dataset):
    if dataset == 'ATLAS':
        transform_mirror = vtk.vtkTransform()
        transform_mirror.Scale(-1, 1, 1)
        mesh.transform(transform_mirror)
        transform_scale = vtk.vtkTransform()
        transform_scale.Scale(1000, 1000, 1000)
        mesh.transform(transform_scale)

def register_mesh(mesh, registration):
    registration = registration.squeeze(0)
    # Create a registration transform for the mesh
    registration_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            registration_matrix.SetElement(i, j, registration[i, j])
    transform_register = vtk.vtkTransform()
    # Applying 4x4 directly
    transform_register.SetMatrix(registration_matrix)
    # Transforms mesh in place
    mesh.transform(transform_register)

def undo_register_mesh(mesh, registration):
    # Invert the registration process
    registration = registration.squeeze(0)
    inverse_registration = np.linalg.inv(registration)
    # Create a registration transform for the mesh
    inverse_registration_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            inverse_registration_matrix.SetElement(i, j, inverse_registration[i, j])
    inverse_transform_register = vtk.vtkTransform()
    inverse_transform_register.SetMatrix(inverse_registration_matrix)
    mesh.transform(inverse_transform_register)