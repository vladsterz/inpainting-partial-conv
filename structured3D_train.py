import numpy as np
import torch
import cv2
from numpy.random import default_rng

class NFOV():
    def __init__(self, height=400, width=800, FOV = [0.45, 0.45]):
        self.FOV = FOV
        pi = 3.1415
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

class DatasetStructure3D(torch.utils.data.Dataset):
    def __init__(self, path, load = None):
        import os
        super().__init__()

        # self.images = [os.path.join(path, x, [y for y in os.listdir(os.path.join(path, x)) if "rgb" in y][0]) for x in os.listdir(path)]
        self.images = [os.path.join(path, x, [y for y in os.listdir(os.path.join(path,x)) if "rgb" in y][0]) for x in os.listdir(path) if "empty" in x]
        self.layout = [os.path.join(os.path.dirname(x), "layout.txt") for x in self.images]

        self.nvfov = NFOV(128,128)
        self.rng = default_rng(seed = 271092)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        layout = np.loadtxt(self.layout[index])
        img = cv2.imread(self.images[index])

        number_of_wall_intersections = layout.shape[0] / 2
        target_corner = int(self.rng.random() * number_of_wall_intersections)
        phi , theta = layout[target_corner * 2 + 1, 0] / img.shape[1] , layout[target_corner * 2 + 1, 1] / img.shape[0]
        try:
            out = self.nvfov.toNFOV(img, np.array([phi, theta]))
        except:
            out = np.zeros((self.nvfov.width,self.nvfov.height,3))

        mask = torch.ones((3, self.nvfov.width,self.nvfov.height))
        width = torch.randint(50, 75, (1,))
        height = torch.randint(50, 75, (1,))
        top_left_w = torch.randint(self.nvfov.width - width.item(), (1,))
        top_left_h = torch.randint(self.nvfov.height - height.item(), (1,))
        mask[:, top_left_w:top_left_w + width, top_left_h: top_left_h + height] = 0

        img = torch.from_numpy(out).permute(2,0,1).float()/255.0
        return  img * mask , mask , img