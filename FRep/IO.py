import numpy as np
import torch


def writeOFF(filename, verts, faces):
    f = open(filename, 'w')

    n_verts = len(verts)
    n_faces = len(faces)

    f.write('OFF\n')
    f.write(str(n_verts) + ' ' + str(n_faces) + ' ' + '0\n')

    for i in range(n_verts):
        v = verts[i]
        if (len(v) != 3):
            print('Invalid vertex: ' + str(i) + '\n')
        f.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')

    for i in range(n_faces):
        face = faces[i]
        if (len(face) != 3):
            print('Invalid triangle: ' + str(i) + ' contains ' + len(face) +
                  ' vertices\n')
        f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) +
                '\n')

    f.close()


def readOFF(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return np.array(vertices), np.array(faces)


def readPointCloud(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f)

    return data


# Save field
def saveVTK(filename, xyz, res, field):
    resx, resy, resz = res

    field_title = 'VALUE'

    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('vtk output\n')
        f.write('ASCII\n')
        f.write('DATASET STRUCTURED_GRID\n')
        f.write('DIMENSIONS ' + str(resx) + ' ' + str(resy) + ' ' + str(resz) +
                '\n')
        f.write('POINTS ' + str(resx * resy * resz) + ' double\n')

        if (torch.is_tensor(xyz)):
            np.savetxt(f, xyz.detach().cpu().numpy())
        else:
            np.savetxt(f, xyz)

        f.write('\n\n')

        f.write('POINT_DATA ' + str(resx * resy * resz) + '\n')
        f.write('SCALARS ' + field_title + ' double' + '\n')
        f.write('LOOKUP_TABLE default\n')

        if (torch.is_tensor(field)):
            np.savetxt(f, field.detach().cpu().numpy())
        else:
            np.savetxt(f, field)
        f.write('\n')
