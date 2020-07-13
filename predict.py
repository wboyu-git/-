import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


# 不能保证包括所有省份
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]