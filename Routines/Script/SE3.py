import numpy as np
np.warnings.filterwarnings('ignore')
import quaternion



def compose_svd_mat4(t, ql, s, qr):
    ql = np.quaternion(*ql)
    qr = np.quaternion(*qr)
    T = np.eye(4)
    T[0:3, 3] = t
    Rl = np.eye(4)
    Rl[0:3, 0:3] = quaternion.as_rotation_matrix(ql)
    Rr = np.eye(4)
    Rr[0:3, 0:3] = quaternion.as_rotation_matrix(qr)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(Rl).dot(S).dot(Rr)
    return M 

def decompose_svd_mat4(M):
    Rl,S,Rr = np.linalg.svd(M[0:3,0:3])

    s = S
    ql = quaternion.from_rotation_matrix(Rl)
    qr = quaternion.from_rotation_matrix(Rr)

    t = M[0:3, 3]
    return t, ql, s, qr

def compose_mat4(t, q, s, center=None):
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M 

def decompose_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])
    #q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s

def invert_mat4(M):
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    M1 = np.eye(4)
    M1[0:3,0:3] = R.transpose()
    M1[0:3,3] = -np.dot(R.transpose(),M[0:3,3])

    return np.dot(np.diag([1.0/sx,1.0/sy,1.0/sz,1.0]),M1)
