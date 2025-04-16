import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import copy
from termcolor import colored
import os

def load_image(path, side_size=512):
    return np.array(Image.open(path).resize((side_size, side_size)).convert('L')).astype(np.int16)

def int2bin(n, length):
    binary = bin(n)[2:].zfill(length)
    return binary

def psnr(old, new):
    if old.size != new.size:
        raise ValueError(f"old.size{old.size} != new.size{new.size}")

    mse = np.mean((old - new) ** 2)
    if mse == 0:
        return float('inf')
    
    max_val = 255.0
    res = 20 * np.log10(max_val / np.sqrt(mse))
    return float(res)

def show_images(*images):
    count = len(images)
    rows = count // 3
    plt.figure(figsize=(5 * 3, 5 * rows))

    row, col = 0, 0
    for i, image in enumerate(images):
        if isinstance(image, tuple):
            img, name = image
        else:
            img, name = image, "unnamed"

        plt.subplot(rows, 3, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'{name}')

    plt.show()

def show_diff(orig, new, mode='combined'):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow((orig * 255).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title(f'orig')

    masked = np.stack([new]*3, axis=-1)
    
    if mode == 'combined':
        diff = new - orig
        red = np.zeros_like(diff, dtype=bool)
        red[diff < 0] = True
        green = np.zeros_like(diff, dtype=bool)
        green[diff > 0] = True
        masked[red, 0] = 1
        masked[green, 2] = 1
    elif mode == 'diff':
        diff = new - orig
        mask = np.zeros_like(new, dtype=bool)
        mask[diff != 0] = True
        masked[mask, 0] = 1
    elif mode == 'disable':
        pass
    else:
        raise ValueError('unknown mode')


    plt.subplot(1, 2, 2)
    plt.imshow((masked * 255).astype(int))
    plt.axis('off')
    plt.title(f'new')

def find_common_divisors(a, b):
    min_value = min(a, b)
    common_divisors = []

    for i in range(1, min_value + 1):
        if a % i == 0 and b % i == 0:
            common_divisors.append(i)

    return common_divisors

def gen_test(size, integer=True, random=True):
    if integer:
        if random:
            return np.random.randint(0, 255, (size, size), dtype=np.int16)
        else:
            sequence = np.linspace(0, 255, size * size, endpoint=True, dtype=np.int16)
            matrix = sequence.reshape((size, size))    
            return matrix
    else:
        if not random:
            sequence = np.linspace(0, 1, size * size, endpoint=True)
            matrix = sequence.reshape((size, size))    
            return matrix
        else:
            return np.random.rand(size, size)
        
def gen_grad(size):
    sequence = np.linspace(0, size * size, size * size, endpoint=False, dtype=np.int16)
    matrix = sequence.reshape((size, size))    
    return matrix

def pmat(mat, pre=''):
    n = len(mat)
    integer = (mat.dtype == np.int16)

    for i in range(n):
        print(f'{pre}', end='')
        for j in range(n):

            if integer:
                val = str(mat[i][j])
                print(f'{val:>5}', end=' ')
            else:
                val = f'{mat[i][j]:.4f}'
                print(f'{val:>10}', end=' ')
        print()
    print()

def mat_equal(left, right):
    return np.max(np.abs(left - right)) <= 1

def mat_max_diff(left, right):
    return np.max(np.abs(left - right))

def check_imgs_equal(orig, restored):
    result = mat_equal(orig, restored)
    label = colored('True', 'green') if result else colored('False', 'red')
    print(f'orig == restored      = {label}')

def pbool(b):
    if b:
        return colored('True', 'green')
    else:
        return colored('False', 'red')

def check_mess_equal(mess, mess_restored):
    if mess == mess_restored:
        print(f'mess == mess_restored = {colored('True', 'green')}')
    else:
        print(f'mess == mess_restored = {colored('False', 'red')}  failed', end='')
        indexes = ''
        for i in range(len(mess)):
            if mess[i] != mess_restored[i]:
                indexes = indexes + f'{i} '
        print(f'({len(indexes)} = {len(indexes) / len(mess) * 100:.2f}%) at: {indexes}')

def gen_mess(l):
    return ''.join(np.random.choice(['0', '1'], size=(l)))

def norm_mess(mess, max=64):
    if len(mess) <= max:
        return mess
    else:
        return mess[:max+1] + '...'
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

def progress(current, total):
    printProgressBar(current, total, prefix = 'Progress:', suffix = 'Complete', length = 50)

def gen_graph(x, y, name_x, name_y, title=None):
    # plt.plot(x, y)
    plt.plot(range(len(x)), y, marker='o')
    plt.xticks(ticks=range(len(x)), labels=x)
    if title is not None:
        plt.title(title)
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.show()

def get_message(version, length):
    if f"{version}" in os.listdir("./inputs/watermarks") and f"{length}.txt" in os.listdir(f"./inputs/watermarks/{version}"):
        with open(f"./inputs/watermarks/{version}/{length}.txt", "r") as f:
            return f.read()
    else:
        mess = gen_mess(length)
        save_message(mess, version, length)
        return mess

def save_message(message, version, length):
    if not os.path.exists(f"./inputs/watermarks/{version}"):
        os.makedirs(f"./inputs/watermarks/{version}")
    with open(f"./inputs/watermarks/{version}/{length}.txt", "w") as f:
        f.write(message)