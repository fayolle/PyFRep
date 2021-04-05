import skimage.measure as sk
import numpy as np


def evalToMesh(sdf, grid_min, grid_max, cell_size):
    nx = int((grid_max[0] - grid_min[0]) / cell_size)
    ny = int((grid_max[1] - grid_min[1]) / cell_size)
    nz = int((grid_max[2] - grid_min[2]) / cell_size)

    x_ = np.linspace(grid_min[0], grid_max[0], nx)
    y_ = np.linspace(grid_min[1], grid_max[1], ny)
    z_ = np.linspace(grid_min[2], grid_max[2], nz)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

    p = np.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), axis=1)

    d = sdf(p)

    volume = d.reshape((nx,ny,nz)).transpose()

    #vertices, faces, normals, _ = sk.marching_cubes_lewiner(volume, level=0)
    vertices, faces, normals, _ = sk.marching_cubes(volume, level=0)

    return vertices, faces, normals


def writeOFF(filename, verts, faces):
    f = open(filename, 'w')

    n_verts = len(verts)
    n_faces = len(faces)

    f.write('OFF\n')
    f.write(str(n_verts)+' '+str(n_faces)+' '+'0\n')

    for i in range(n_verts):
        v = verts[i]
        if (len(v)!=3):
            print('Invalid vertex: '+str(i)+'\n')
        f.write(str(v[0])+' '+str(v[1])+' '+str(v[2])+'\n')

    for i in range(n_faces):
        face = faces[i]
        if (len(face)!=3):
            print('Invalid triangle: '+str(i)+' contains '+len(face)+' vertices\n')
        f.write('3 '+str(face[0])+' '+str(face[1])+' '+str(face[2])+'\n')

    f.close()


def writeMesh(filename, model, grid_min, grid_max, cell_size):
    verts, faces, normals = evalToMesh(model, grid_min, grid_max, cell_size)
    writeOFF(filename, verts, faces)


