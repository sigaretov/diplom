import numpy as np
from matplotlib import pyplot as plt
import math
import utils as ut

def split2blocks(arr, size):
    rows, cols = arr.shape
    row_blocks = rows // size
    col_blocks = cols // size
    reshaped = arr.reshape(row_blocks, size, col_blocks, size)
    blocks = reshaped.swapaxes(1, 2)    
    return blocks


def show_hl(blocks, name=""):
    if len(blocks) != 4:
        raise ValueError(f"Expected LL, HL, LH, HH blocks")
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].imshow(blocks[i * 2 + j], cmap='gray', vmin=0, vmax=255)
            axs[i, j].axis('off')
    
    plt.show()


def iwt1(row):
    n = len(row)

    if n % 2 != 0:
        raise ValueError(f"Row size % 2 != 0")
    
    low = np.empty(n // 2)
    high = np.empty(n // 2)
    for i in range(n // 2):
        low[i] = (row[2 * i + 1] + row[2 * i]) // 2
        high[i] = row[2 * i + 1] - row[2 * i]

    # print(f"iwt1: {row=} {low=} {high=}\n")
    
    return low, high


def inv_iwt1(low, high):
    l = len(low)
    
    if low.shape != high.shape:
        raise ValueError(f"Low shape != high shape")
    
    res = np.empty(l * 2)
    for i in range(l):
        res[2 * i] = low[i] - high[i] // 2
        res[2 * i + 1] = low[i] + (high[i] + 1) // 2
    return res


def iwt2(img):
    n = len(img)

    if n != len(img[0]):
        raise ValueError(f"Image heigth != width")
    if n % 2 != 0:
        raise ValueError(f"Image size % 2 != 0")
    
    side = n // 2
    low = np.empty((n, side))
    high = np.empty((n, side))


    for i in range(n):
        low[i], high[i] = iwt1(img[i])

    # print(f"iwt2: {low=}")
    # print(f"iwt2: {high=}\n")

    ll = np.empty((side, side))
    hl = np.empty((side, side))
    lh = np.empty((side, side))
    hh = np.empty((side, side))

    for i in range(n // 2):
        ll[:, i], lh[:, i] = iwt1(low[:, i])
        hl[:, i], hh[:, i] = iwt1(high[:, i])

    return [ll, hl, lh, hh]
    

def inv_iwt2(ll, hl, lh, hh):
    side = max(max(ll.shape), max(hl.shape), max(lh.shape), max(hh.shape))

    if side != min(min(ll.shape), min(hl.shape), min(lh.shape), min(hh.shape)):
        raise ValueError(f"Sides are not equal")

    low = np.empty((side * 2, side))
    high = np.empty((side * 2, side))

    for i in range(side):
        low[:, i] = inv_iwt1(ll[:, i], lh[:, i])
        high[:, i] = inv_iwt1(hl[:, i], hh[:, i])
    
    res = np.zeros((side * 2, side * 2))
    for i in range(side * 2):
        res[i] = inv_iwt1(low[i], high[i])
    
    return res.astype(dtype=np.int16)


def test_iwt1():
    size = 100
    for i in range(100):
        orig = np.random.randint(0, 255, size)
        l, h = iwt1(orig)
        new = inv_iwt1(l, h)
        if not np.array_equal(orig, new):
            print("failed iwt1", orig.shape, new.shape)
            print(orig)
            print(new)


def test_iwt2():
    size = 100
    for i in range(100):
        orig = np.random.randint(0, 255, (size, size))

        ll, hl, lh, hh = iwt2(orig)

        # print(f"{ll=} {hl=}")
        # print(f"{lh=} {hh=}\n")
        
        new = inv_iwt2(ll, hl, lh, hh)

        if not np.array_equal(orig, new):
            print("\nFAILED")
            print(f"{orig=}\n")
            print(f"{new=}\n")

def gen_slant_matrix(n):
    S = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    a = 1

    for i in range(2, n + 1):
        b = 1 / np.sqrt(1 + 4 * a**2)
        a = 2 * b * a

        q1 = np.array([[1, 0], [a, b]])
        q2 = np.array([[1, 0], [-a, b]])
        q3 = np.array([[0, 1], [-b, a]])
        q4 = np.array([[0, -1], [b, a]])

        Z = np.concatenate([np.concatenate([S, np.zeros(S.shape)], axis=1),
                 np.concatenate([np.zeros(S.shape), S], axis=1)])

        if (i == 2):
            B1 = np.concatenate([q1, q2], axis=1)  # block 1
            B2 = np.concatenate([q3, q4], axis=1)  # block 2
            S = (1 / np.sqrt(2)) * np.concatenate([B1, B2]) @ Z

        else:
            k = int((2**i - 4) / 2)
            B1 = np.concatenate([q1, np.zeros([2, k]), q2, np.zeros([2, k])], axis=1)  # block 1
            B2 = np.concatenate([np.zeros([k, 2]), np.eye(k), np.zeros(
                [k, 2]), np.eye(k)], axis=1)  # block 2
            B3 = np.concatenate([q3, np.zeros([2, k]), q4, np.zeros([2, k])], axis=1)  # block 3
            B4 = np.concatenate([np.zeros([k, 2]), np.eye(k), np.zeros(
                [k, 2]), -np.eye(k)], axis=1)  # block 4

            S = (1 / np.sqrt(2)) * np.concatenate([B1, B2, B3, B4]) @ Z

    return S

def mat_str(mat):
    res = ""
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            res = res + str(mat[i][j]) + ' '
        res = res + '\n'
    return res

def apply_slt(orig):
    side = orig.shape[0]
    mat_power = int(math.log2(side))

    if mat_power < 2 or mat_power > 10:
        raise ValueError(f'shape: {orig.shape}, mat_power is {mat_power}')

    mat = np.load(f'./matrixes/mat-{mat_power}.npy')
    new = mat @ orig @ np.transpose(mat)
    return new

def apply_inverse_slt(orig, round=False):
    side = orig.shape[0]
    mat_power = int(math.log2(side))

    if mat_power < 2 or mat_power > 10:
        raise ValueError(f'shape: {orig.shape}, mat_power is {mat_power}')

    mat = np.load(f'./matrixes/mat-{mat_power}.npy')
    new = np.transpose(mat) @ orig @ mat

    if round:
        new = np.around(new).astype(int)
    return new