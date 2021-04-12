import skimage.measure as sk
import numpy as np
import torch
from .grid import torchGrid, torchSampling


def evalToMesh(model, grid_min, grid_max, grid_res, device='cpu'):
    x, y, z = torchGrid(grid_min, grid_max, grid_res, device)
    volume = torchSampling(model, x, y, z)
    volume = volume.detach().cpu().numpy()

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


def writeMesh(filename, model, grid_min, grid_max, grid_res, device='cpu'):
    verts, faces, normals = evalToMesh(model, grid_min, grid_max, grid_res, device)
    writeOFF(filename, verts, faces)


