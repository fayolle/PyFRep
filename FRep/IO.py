import numpy as np
import torch
from pathlib import Path


class FileFormatError(Exception):
    """Custom exception for file format errors"""
    pass


class InvalidDataError(Exception):
    """Custom exception for invalid data"""
    pass


def writeOFF(filename, verts, faces):
    if verts is None or faces is None:
        raise InvalidDataError("Vertices and faces cannot be None")
    
    try:
        verts = np.asarray(verts)
        faces = np.asarray(faces)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Vertices and faces must be array-like: {e}")
    
    if len(verts.shape) != 2 or verts.shape[1] != 3:
        raise InvalidDataError(
            f"Vertices must have shape (N, 3), got {verts.shape}"
        )
    
    if len(faces.shape) != 2 or faces.shape[1] != 3:
        raise InvalidDataError(
            f"Faces must have shape (M, 3), got {faces.shape}"
        )
    
    n_verts = len(verts)
    n_faces = len(faces)
    
    if n_faces > 0:
        if faces.min() < 0 or faces.max() >= n_verts:
            raise InvalidDataError(
                f"Face indices must be in range [0, {n_verts-1}]"
            )
    
    try:
        with open(filename, 'w') as f:
            f.write('OFF\n')
            f.write(f'{n_verts} {n_faces} 0\n')
            
            for v in verts:
                f.write(f'{v[0]} {v[1]} {v[2]}\n')
            
            for face in faces:
                f.write(f'3 {face[0]} {face[1]} {face[2]}\n')
    except IOError as e:
        raise IOError(f"Failed to write OFF file '{filename}': {e}")


def readOFF(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        with open(filename, 'r') as fp:
            lines = fp.readlines()
    except IOError as e:
        raise IOError(f"Failed to read file '{filename}': {e}")
    
    if not lines:
        raise FileFormatError("File is empty")
    
    lines = [line.strip() for line in lines if line.strip()]
    
    if len(lines) < 2:
        raise FileFormatError("File has insufficient lines")
    
    if lines[0] != 'OFF':
        raise FileFormatError(f"Expected 'OFF' header, got '{lines[0]}'")
    
    try:
        parts = lines[1].split()
        if len(parts) < 2:
            raise FileFormatError(f"Invalid header line: {lines[1]}")
        
        num_vertices = int(parts[0])
        num_faces = int(parts[1])
    except (ValueError, IndexError) as e:
        raise FileFormatError(f"Invalid header format: {e}")
    
    if num_vertices <= 0:
        raise FileFormatError(f"Invalid vertex count: {num_vertices}")
    if num_faces < 0:
        raise FileFormatError(f"Invalid face count: {num_faces}")
    
    expected_lines = 2 + num_vertices + num_faces
    if len(lines) < expected_lines:
        raise FileFormatError(
            f"Expected {expected_lines} lines, got {len(lines)}"
        )
    
    vertices = []
    try:
        for i in range(num_vertices):
            vertex = lines[2 + i].split()
            if len(vertex) < 3:
                raise FileFormatError(
                    f"Vertex {i} has {len(vertex)} coordinates, expected 3"
                )
            vertex = [float(point) for point in vertex[:3]]
            vertices.append(vertex)
    except (ValueError, IndexError) as e:
        raise FileFormatError(f"Error parsing vertex {i}: {e}")
    
    faces = []
    try:
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split()
            if len(face) < 1:
                raise FileFormatError(f"Face {i} is empty")
            
            face = [int(index) for index in face]
            
            face_size = face[0]
            if face_size != len(face) - 1:
                raise FileFormatError(
                    f"Face {i}: size mismatch ({face_size} vs {len(face)-1})"
                )
            
            for idx in face[1:]:
                if idx < 0 or idx >= num_vertices:
                    raise FileFormatError(
                        f"Face {i}: invalid vertex index {idx}"
                    )
            
            faces.append(face[1:])
    except (ValueError, IndexError) as e:
        raise FileFormatError(f"Error parsing face {i}: {e}")
    
    return np.array(vertices), np.array(faces)


def readPointCloud(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        with open(filename, 'r') as f:
            data = np.loadtxt(f)
    except IOError as e:
        raise IOError(f"Failed to read file '{filename}': {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data format in '{filename}': {e}")
    
    if data.size == 0:
        raise ValueError("File contains no data")
    
    return data


def writeVTK(filename, xyz, res, field):
    if len(res) != 3:
        raise InvalidDataError(f"Resolution must be 3-tuple, got {len(res)}")
    
    resx, resy, resz = res
    expected_points = resx * resy * resz
    
    if torch.is_tensor(xyz):
        xyz_data = xyz.detach().cpu().numpy()
    else:
        xyz_data = np.asarray(xyz)
    
    if torch.is_tensor(field):
        field_data = field.detach().cpu().numpy()
    else:
        field_data = np.asarray(field)
    
    if xyz_data.shape[0] != expected_points:
        raise InvalidDataError(
            f"xyz has {xyz_data.shape[0]} points, expected {expected_points}"
        )
    
    if field_data.shape[0] != expected_points:
        raise InvalidDataError(
            f"field has {field_data.shape[0]} values, expected {expected_points}"
        )
    
    try:
        with open(filename, 'w') as f:
            f.write('# vtk DataFile Version 3.0\n')
            f.write('vtk output\n')
            f.write('ASCII\n')
            f.write('DATASET STRUCTURED_GRID\n')
            f.write(f'DIMENSIONS {resx} {resy} {resz}\n')
            f.write(f'POINTS {expected_points} double\n')
            
            np.savetxt(f, xyz_data)
            
            f.write('\n\n')
            f.write(f'POINT_DATA {expected_points}\n')
            f.write('SCALARS VALUE double\n')
            f.write('LOOKUP_TABLE default\n')
            
            np.savetxt(f, field_data)
            f.write('\n')
    except IOError as e:
        raise IOError(f"Failed to write VTK file '{filename}': {e}")

    
# Save mesh with a scalar value per node as VTK
def writeSurfaceMeshVTK(filename, V, F, field):
    V = np.asarray(V)
    F = np.asarray(F)
    field = np.asarray(field)
    
    if len(V.shape) != 2 or V.shape[1] != 3:
        raise InvalidDataError(f"V must have shape (N, 3), got {V.shape}")
    
    if len(F.shape) != 2 or F.shape[1] != 3:
        raise InvalidDataError(f"F must have shape (M, 3), got {F.shape}")
    
    number_nodes = len(V)
    number_elements = len(F)
    
    if field.shape[0] != number_nodes:
        raise InvalidDataError(
            f"field has {field.shape[0]} values, expected {number_nodes}"
        )
    
    if number_elements > 0:
        if F.min() < 0 or F.max() >= number_nodes:
            raise InvalidDataError("Face indices out of bounds")
    
    element_order = 3
    cell_size = number_elements * (element_order + 1)
    
    try:
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Field\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {number_nodes} double\n")

            # Write vertex coordinates
            np.savetxt(f, V, fmt="%f %f %f")
            
            f.write(f"\nCELLS {number_elements} {cell_size}\n")
            for i in range(number_elements):
                f.write(f" {element_order}")
                for j in range(element_order):
                    f.write(f" {F[i, j]}")
                f.write("\n")
            
            f.write(f"\nCELL_TYPES {number_elements}\n")
            # Triangle is cell type 5
            for i in range(number_elements):
                f.write("5\n")
            
            f.write(f"\nPOINT_DATA {number_nodes}\n")
            f.write("SCALARS field double\n")
            f.write("LOOKUP_TABLE default\n")
            
            for i in range(number_nodes):
                f.write(f"{field[i]}\n")
            
            f.write("\n")
    except IOError as e:
        raise IOError(f"Failed to write VTK file '{filename}': {e}")


def writePointCloudVTK(filename, V, field):
    """
    Write point cloud to VTK unstructured grid format.
    
    Args:
        filename: Path to output file
        V: Vertices array of shape (N, 3)
        field: Scalar field values per vertex of shape (N,)
        
    Raises:
        InvalidDataError: If data dimensions are invalid
        IOError: If file cannot be written
    """
    V = np.asarray(V)
    field = np.asarray(field)
    
    if len(V.shape) != 2 or V.shape[1] != 3:
        raise InvalidDataError(f"V must have shape (N, 3), got {V.shape}")
    
    number_nodes = len(V)
    
    if field.shape[0] != number_nodes:
        raise InvalidDataError(
            f"field has {field.shape[0]} values, expected {number_nodes}"
        )
    
    try:
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Field\n")
            f.write("ASCII\n\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {number_nodes} double\n")
            
            np.savetxt(f, V, fmt="%f %f %f")
            
            f.write(f"\nCELLS {number_nodes} {2*number_nodes}\n")
            for i in range(number_nodes):
                f.write(f" 1 {i}\n")
            
            f.write(f"\nCELL_TYPES {number_nodes}\n")
            # Vertex is cell type 1
            for i in range(number_nodes):
                f.write("1\n")
            
            f.write(f"\nPOINT_DATA {number_nodes}\n")
            f.write("SCALARS field double\n")
            f.write("LOOKUP_TABLE default\n")
            
            for i in range(number_nodes):
                f.write(f"{field[i]}\n")
            
            f.write("\n")
    except IOError as e:
        raise IOError(f"Failed to write VTK file '{filename}': {e}")
