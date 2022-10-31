import FRep
from FRep.sdf_primitives import *
from FRep.sdf_ops import *
from FRep.mesh import *
from FRep.ops import rotate3DX

import polyscope as ps


def model(p):
    piover2 = 1.5707963267948966192
    p = rotate3DX(p, -piover2)

    a1 = [1.078, 3.327, -3.0]
    b1 = [1.078, 3.327, 3.0]
    cyl1 = cappedCylinder(p, a1, b1, 0.5)

    a2 = [0.0, 0.0, -3.0]
    b2 = [0.0, 0.0, 0.0]
    cyl2 = cappedCylinder(p, a2, b2, 5.0)
    
    a3 = [3.5, 0.0, 0.0]
    b3 = [3.5, 0.0, 3.0]
    cyl3 = cappedCylinder(p, a3, b3, 0.5)

    a4 = [-1.075, -3.329, -19.0]
    b4 = [-1.075, -3.329, 19.0]
    cyl4 = cappedCylinder(p, a4, b4, 1.0)

    a5 = [-1.081, 3.325, -19.0]
    b5 = [-1.081, 3.325, 19.0]
    cyl5 = cappedCylinder(p, a5, b5, 1.0)

    a6 = [1.081, -3.326, 0.0]
    b6 = [1.081, -3.326, 3.0]
    cyl6 = cappedCylinder(p, a6, b6, 0.5)

    a7 = [0.03, 0.015, 0.0]
    b7 = [0.03, 0.015, 2.5]
    cyl7 = cappedCylinder(p, a7, b7, 2.0)

    a8 = [0.0, 0.0, -3.0]
    b8 = [0.0, 0.0, 2.5]
    cyl8 = cappedCylinder(p, a8, b8, 1.5)

    a9 = [2.837, -2.072, -3.0]
    b9 = [2.837, -2.072, 19.0]
    cyl9 = cappedCylinder(p, a9, b9, 1.0)

    a10 = [-3.496, -0.013, -3.0]
    b10 = [-3.496, -0.013, 19.0]
    cyl10 = cappedCylinder(p, a10, b10, 1.0)

    a11 = [2.83, 2.059, -3.0]
    b11 = [2.83, 2.059, 0.0]
    cyl11 = cappedCylinder(p, a11, b11, 1.0)

    a12 = [-2.835, -2.058, 0.0]
    b12 = [-2.835, -2.058, 3.0]
    cyl12 = cappedCylinder(p, a12, b12, 0.5)

    a13 = [-2.834, 2.051, 0.0]
    b13 = [-2.834, 2.051, 3.0]
    cyl13 = cappedCylinder(p, a13, b13, 0.5)

    tmp1 = union(cyl2, cyl3)
    tmp2 = union(cyl4, cyl5)
    tmp3 = difference(tmp1, tmp2)
    tmp4 = union(cyl1, tmp3)
    tmp5 = union(tmp4, cyl6)
    tmp6 = union(tmp5, cyl7)

    tmp7 = union(cyl8, cyl9)
    tmp8 = union(tmp7, cyl10)
    tmp9 = union(tmp8, cyl11)

    tmp10 = difference(tmp6, tmp9)

    tmp11 = union(tmp10, cyl12)
    tmp12 = union(tmp11, cyl13)

    return tmp12


v, f, n = evalToMesh(model,
                     grid_min=(-6.0, -3.6, -6.0),
                     grid_max=(6.0, 3.6, 6.0),
                     grid_res=(128, 128, 128))


ps.init()
ps.register_surface_mesh("mesh", v, f)
ps.show()
