"""Motion primitives class."""
import numpy as np
from misc import loadmat
from os.path import splitext
import pickle
import matplotlib.pyplot as plt
import casadi as ca
from copy import deepcopy


class MotionPrimitives:
    def __init__(self, filename=''):
        if len(filename) > 0:
            self.load(filename)

    def load(self, filename):
        ext = splitext(filename)[-1]
        if ext == '.mat':
            mprims = loadmat(filename)['mprims']

            self.mprims = []
            for i in range(len(mprims)):
                mi = []
                for j in range(len(mprims[0])):
                    mi_element = {'x': mprims[i][j].x, 'y': mprims[i][j].y,
                                  'u': mprims[i][j].u, 'th': mprims[i][j].th,
                                  'T': mprims[i][j].T, 'ds': mprims[i][j].ds}
                    mi.append(mi_element)
                self.mprims.append(mi)
            self.mprims = np.array(self.mprims)

            self.th = np.array([mi[0].th[0] for mi in mprims])
        elif ext == '.pickle':
            with open(filename, 'rb') as f:
                (self.mprims, self.th) = pickle.load(f)
        else:
            raise Exception('Unknown file type, only .mat supported')

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.mprims, self.th), f)

    def plot(self, *args, **kwargs):
        for mp in self.mprims:
            for mpi in mp:
                plt.plot(mpi['x'], mpi['y'], *args, **kwargs)

    def PlanToPath(self, x0, plan):
        p = np.array([x0])
        for ui in plan['control']:
            mpi = self.mprims[ui[0], ui[1]]
            traj = p[-1] + np.column_stack((mpi['x'], mpi['y'], mpi['th']))
            p = np.vstack((p, traj))
        return p

    def control_to_path(self, x0, control):
        p = np.array([x0])
        for ui in control:
            mpi = self.mprims[ui[0], ui[1]]
            traj = p[-1] + np.column_stack((mpi['x'], mpi['y'], mpi['th']))
            p = np.vstack((p, traj))
        return p

    
    def generate_primitives(self, theta_init, state_0, L=2, v=10,
                                 u_max=np.pi/4, print_level=3):
        self.mprims = compute_motion_primitives(theta_init, state_0, L, v,
                                                u_max, print_level)
        self.th = np.array([mi[0]['th'][0] for mi in self.mprims])


def compute_motion_primitives(theta_init, state_0, L, v, u_max, print_level=3):
    def compute_mprim(state_i, lattice, L, v, u_max, print_level):
        N = 75
        nx = 3
        Nc = 3
        mprim = []
        for state_f in lattice:
            x = ca.MX.sym('x', nx)
            u = ca.MX.sym('u')

            F = ca.Function('f', [x, u],
                            [v*ca.cos(x[2]), v*ca.sin(x[2]), v*ca.tan(u)/L])

            opti = ca.Opti()
            X = opti.variable(nx, N+1)
            pos_x = X[0, :]
            pos_y = X[1, :]
            ang_th = X[2, :]
            U = opti.variable(N, 1)
            T = opti.variable(1)
            dt = T/N

            opti.set_initial(T, 0.1)
            opti.set_initial(U, 0.0*np.ones(N))
            opti.set_initial(pos_x, np.linspace(state_i[0], state_f[0], N+1))
            opti.set_initial(pos_y, np.linspace(state_i[1], state_f[1], N+1))

            tau = ca.collocation_points(Nc, 'radau')
            C, _ = ca.collocation_interpolators(tau)

            Xc_vec = []
            for k in range(N):
                Xc = opti.variable(nx, Nc)
                Xc_vec.append(Xc)
                X_kc = ca.horzcat(X[:, k], Xc)
                for j in range(Nc):
                    fo = F(Xc[:, j], U[k])
                    opti.subject_to(X_kc@C[j+1] == dt*ca.vertcat(fo[0], fo[1], fo[2]))

                opti.subject_to(X_kc[:, Nc] == X[:, k+1]) 

            for k in range(N):
                opti.subject_to(U[k] <= u_max)
                opti.subject_to(-u_max <= U[k])

            opti.subject_to(T >= 0.001)
            opti.subject_to(X[:, 0] == state_i)
            opti.subject_to(X[:, -1] == state_f)

            alpha = 1e-2
            opti.minimize(T + alpha*ca.sumsqr(U))

            opti.solver('ipopt', {'expand': True},
                        {'tol': 10**-8, 'print_level': print_level})
            sol = opti.solve()

            pos_x_opt = sol.value(pos_x)
            pos_y_opt = sol.value(pos_y)
            ang_th_opt = sol.value(ang_th)
            u_opt = sol.value(U)
            T_opt = sol.value(T)
            mprim.append({'x': pos_x_opt, 'y': pos_y_opt, 'th': ang_th_opt,
                          'u': u_opt, 'T': T_opt, 'ds': T_opt*v})
        return np.array(mprim)
    N = len(theta_init)
    dth = 2*np.pi/N

    mprims = []
    state_i = np.array([0, 0, theta_init[0]])
    mprims.append(compute_mprim(state_i, state_0, L, v, u_max, print_level))

    xy_pi_4 = np.sqrt(2)*(
        state_0[:, 0:2]@np.array([[np.cos(dth), np.sin(dth)],
                                  [-np.sin(dth), np.cos(dth)]]))
    state_pi_4 = np.column_stack((xy_pi_4, state_0[:, 2] + np.pi/4))
    state_i = np.array([0, 0, theta_init[1]])
    mprims.append(compute_mprim(state_i, state_pi_4, L, v, u_max, print_level))

    for i in range(2, len(theta_init)):
        if i % 2 == 1:
            mprims.append(deepcopy(mprims[1]))
            th = dth*i-np.pi/4
        else:
            mprims.append(deepcopy(mprims[0]))
            th = dth*i

        for j in range(len(mprims[i])):
            mp = mprims[i][j]
            xy_vec_rot = np.array(
                [[np.cos(th), -np.sin(th)],
                 [np.sin(th), np.cos(th)]])@np.vstack((mp['x'], mp['y']))
            mprims[i][j]['x'] = xy_vec_rot[0]
            mprims[i][j]['y'] = xy_vec_rot[1]
            mprims[i][j]['th'] += th
            mprims[i][j]['th'] -= (mprims[i][j]['th'] > np.pi)*np.pi*2
            mprims[i][j]['th'] += (mprims[i][j]['th'] < -np.pi)*np.pi*2

    return np.array(mprims)
