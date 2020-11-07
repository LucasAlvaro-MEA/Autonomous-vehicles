"""Class to represent the planning world"""
import numpy as np
import matplotlib.pyplot as plt

class BoxWorld:
    def __init__(self, lattice):
        self.st_sp = state_space_from_lattice(lattice)
        self.xmin = np.min(lattice[0])
        self.xmax = np.max(lattice[0])
        self.ymin = np.min(lattice[1])
        self.ymax = np.max(lattice[1])
    
        self._fig = None
        self._boxes = []
        self.x_obst = np.array([]).reshape((0, 2))
        self.y_obst = np.array([]).reshape((0, 2))

    def num_nodes(self):
        return self.st_sp.shape[1]
        
    def add_box(self, x, y, W1, W2, fill_box=True):
        self._boxes.append((x, y, W1, W2, fill_box))
        self.x_obst = np.row_stack((self.x_obst, [x, x + W1]))
        self.y_obst = np.row_stack((self.y_obst, [y, y + W2]))

    def draw_box(self, b, *args, **kwargs):
        x0, y0, W1, W2, fill_box = b

        if fill_box:
            plt.fill([x0, x0 + W1, x0 + W1, x0, x0],
                     [y0, y0, y0 + W2, y0 + W2, y0], *args, **kwargs)
        else:
            plt.plot([x0, x0 + W1, x0 + W1, x0, x0],
                     [y0, y0, y0 + W2, y0 + W2, y0], *args, **kwargs)

    def register_figure(self, fig):
        self._fig = fig

    def draw(self, *args, **kwargs):
        if len(args) == 0:
            args = ['r']
        if len(kwargs) == 0:
            kwargs = {'edgecolor': 'k'}

        self.redraw_boxes(*args, **kwargs)

    def redraw_boxes(self, *args, **kwargs):
        for bi in self._boxes:
            self.draw_box(bi, *args, **kwargs)
        if self._fig:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def in_bound(self, point):
        c = False
        if (point[0] >= self.xmin) and (point[0] <= self.xmax) and (point[1] >= self.ymin) and (point[1] <= self.ymax):
            c = True
        return c
            
    def ObstacleFree(self, p):
        for ii in range(p.shape[1]):
            if obstacle_check(p[0, ii], p[1, ii], self.x_obst, self.y_obst):
                return False
        return True

    def ObstacleFree2(self, p):
        for bi in self._boxes:
            xmin, ymin, W1, W2, _ = bi
            xmax = xmin + W1
            ymax = ymin + W2
            if np.any(np.logical_and(np.logical_and(p[0] > xmin,
                                                    p[0] < xmax),
                                     np.logical_and(p[1] > ymin,
                                                    p[1] < ymax))):
                return False
        return True

def state_space_from_lattice(lattice):
    if len(lattice) == 1:
        st_sp = np.array(lattice[0]).reshape((1, -1))
    else:
        st_sp_1 = state_space_from_lattice(lattice[1:])
        N = st_sp_1.shape[1]
        st_sp = np.array([]).reshape((st_sp_1.shape[0] + 1, 0))
        for xi in lattice[0]:
            st_sp = np.hstack((st_sp, 
                               np.row_stack((np.full((1, N), xi),
                                             st_sp_1))))
    return st_sp

def obstacle_check(x, y, x_obst, y_obst):
    for ii in range(x_obst.shape[0]):
        if (x > x_obst[ii, 0] and x < x_obst[ii, 1]) and \
           (y > y_obst[ii, 0] and y < y_obst[ii, 1]):
            return True
    return False
