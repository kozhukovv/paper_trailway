
from typing import List
import numpy as np
import math


# Угол между векторами -PI...+PI
def Angle(v1, v2):
    alpha = math.asin(np.cross(v1, v2)/math.sqrt(np.dot(v1, v1)*np.dot(v2, v2)))
    if np.dot(v1, v2) < 0:
        if alpha > 0:
            alpha = math.pi - alpha
        else:
            alpha = -math.pi - alpha
    return alpha


# Пернендикуляр к отрезку ((x1, y1), (x2,y2)), проходящий через (x1, y1)
def normal(x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    an = b
    bn = -a
    cn = a*y1 - b*x1
    return an, bn, cn


# Точка пресечения прямых a1*x + b1*y + c1 = 0 и a2*x + b2*y + c2 = 0
def cross_lines(a1, b1, c1, a2, b2, c2):
    d = a1*b2 - a2*b1
    x = (b1*c2 - b2*c1)/d
    y = (a2*c1 - a1*c2)/d
    return x, y


# Параметры окружности, соединяющей отрезки ((xb1, yb1), (xe1, ye1)) и  ((xb2, yb2), (xe2, ye2))
def make_circle(xb1, yb1, xe1, ye1, xb2, yb2, xe2, ye2):
    a1, b1, c1 = normal(xe1, ye1, xb1, yb1)
    a2, b2, c2 = normal(xb2, yb2, xe2, ye2)
    xc, yc = cross_lines(a1, b1, c1, a2, b2, c2)
    s = [xe1 - xc, ye1 - yc]
    f = [xb2 - xc, yb2 - yc]
    radius = math.sqrt(np.dot(s, s))
    sector_size = Angle(s, f)
    phi_start = Angle([1., 0.], s)
    return radius, xc, yc, phi_start, sector_size


class Arcspline:
    def __init__(self, data):
        self._data = data
        [r, _] = self._data.shape
        self._data[['radius', 'xc', 'yc', 'phi_start', 'sector_size']] = 0
        self._r = r
        self.spline_data = []

    def build(self):
        r = self._r
        xb = self._data.x_begin
        yb = self._data.y_begin
        xe = self._data.x_end
        ye = self._data.y_end
        # получаем координаты отрезков для построения арочных сегментов
        for idx in range(0, r - 1):
            # можно просто по индексу заполнить колонки
            radius, xc, yc, phi_start, sector_size = make_circle(xb[idx], yb[idx], xe[idx], ye[idx], xb[idx + 1], yb[idx + 1], xe[idx + 1], ye[idx + 1])
            self._data.loc[idx, 'radius'] = radius
            self._data.loc[idx, 'xc'] = xc
            self._data.loc[idx, 'yc'] = yc
            self._data.loc[idx, 'phi_start'] = phi_start
            self._data.loc[idx, 'sector_size'] = sector_size

    def getLines(self):
        dx = self._data.x_begin - self._data.x_end
        dy = self._data.y_begin - self._data.y_end
        ds_line = np.sqrt(dx**2 + dy**2)
        ds_circ = self._data.radius * self._data.sector_size
        ds_circ = ds_circ.abs()
        return ds_line, ds_circ

    def getL(self):
        ds_l, ds_c = self.getLines()
        self.ds_l = ds_l
        self.ds_c = ds_c
        return ds_l.sum() + ds_c.sum()

    def getPoint(self, path: float):
        ds_c = self.ds_c
        ds_l = self.ds_l
        s = 0
        for idx in range(0, len(ds_l)):
            row = self._data.loc[idx]
            # находимся на линейном сегменте
            if path >= s and path < s + ds_l[idx]:
                L = ds_l[idx]
                s_ = path - s
                x = (row.x_end - row.x_begin) * s_/L + row.x_begin
                y = (row.y_end - row.y_begin) * s_/L + row.y_begin
                return x, y
            s = s + ds_l[idx]
            # находимся на арочном сегменте
            if path >= s and path < s + ds_c[idx]:
                if row.radius > 0:
                    sz = row.sector_size
                    radius = row.radius
                    s_ = path - s
                    k = s_/ds_c[idx]
                    alpha = row.phi_start + sz * k
                    x = row.xc + radius * np.cos(alpha)
                    y = row.yc + radius * np.sin(alpha)
                return x, y
            s = s + ds_c[idx]
        return -1000, -1000

    def evalute(self, deg_step=0.5):
        x_basic = list()
        y_basic = list()
        for idx in range(0, self._r):
            xb = self._data.x_begin[idx]
            xe = self._data.x_end[idx]
            yb = self._data.y_begin[idx]
            ye = self._data.y_end[idx]
            ds = np.sqrt((xb - xe)**2 + (yb - ye)**2)
            x_line = np.linspace(xb, xe, int(ds))
            y_line = np.linspace(yb, ye, int(ds))
            x_basic.extend(x_line)
            y_basic.extend(y_line)
            radius = self._data.radius[idx]
            if radius > 0:
                sz = self._data.sector_size[idx]
                n = int(abs(sz) / deg_step)
                phi_0 = self._data.phi_start[idx]
                alpha = np.linspace(phi_0, phi_0 + sz, n)
                x = self._data.xc[idx] + radius * np.cos(alpha[1:])
                y = self._data.yc[idx] + radius * np.sin(alpha[1:])
                x_basic.extend(x)
                y_basic.extend(y)
        self.spline_data = [x_basic, y_basic]
        return x_basic, y_basic

    # получить enOrigin
    def enOrigin(self):
        return self._data.eOrigin[1], self._data.nOrigin[0]

    def getPath(self, yp, xp):
        if len(self.spline_data) == 2:
            x, y = self.spline_data
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            ds = np.sqrt((x - xp)**2 + (y - yp)**2)
            value = ds.min()
            if (value < 100.0):
                idx = np.where(ds == value)[0][0]
                x_short = x[:idx]
                y_short = y[:idx]
                dx = x_short[:-1] - x_short[1:]
                dy = y_short[:-1] - y_short[1:]
                ds_short = np.sqrt(dx**2 + dy**2)
                return ds_short.sum()
            else:
                return -1.0

    def thin(self, length):
        x, y = self.spline_data
        x = np.array(x)
        y = np.array(y)
        s = 0
        result = np.zeros([len(x), 3])
        for i in range(0, len(x)):
            path_ = self.getPath(y[i], x[i])
            if path_ > s:
                result[i] = [x[i], y[i], s]
                s = s + length
        return result[result[:, 1] != 0]