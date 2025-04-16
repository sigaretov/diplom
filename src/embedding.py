import numpy as np
import copy
import transform as tr
import utils as ut
import time
from termcolor import colored
import zlib

class ProcessParams:
    def __init__(self):
        self.debug = False
        self.show = False
        self.use_new_error_function = False
        self.use_new_bit_extraction = False
        self.use_new_side_info = False
        self.compress_side_info = False
    
    def __repr__(self):
        s = ''
        s = s + f'debug                  : {ut.pbool(self.debug)}\n'
        s = s + f'show                   : {ut.pbool(self.show)}\n'
        s = s + f'use new error function : {ut.pbool(self.use_new_error_function)}\n'
        s = s + f'use new bit extraction : {ut.pbool(self.use_new_bit_extraction)}\n'
        s = s + f'use new side info     : {ut.pbool(self.use_new_side_info)}\n'
        s = s + f'compress side info    : {ut.pbool(self.compress_side_info)}\n'
        return s

def split_to_blocks(img, block_size):
    side = img.shape[0]
    if side != img.shape[1]:
        raise ValueError("Image is not square")
    if side % block_size != 0:
        raise ValueError("Image side % block_size != 0")
    
    block_num = side // block_size
    result = np.zeros((block_num, block_num, block_size, block_size), dtype=np.int16)
    
    for i in range(block_num):
        for j in range(block_num):
            y, x = i * block_size, j * block_size
            result[i, j] = img[y : y + block_size, x : x + block_size]

    return result

def combine_blocks(blocks):
    block_num = blocks.shape[0]
    block_len = blocks[0, 0].shape[0]
    res = np.zeros((block_num * block_len, block_num * block_len))
    
    for i in range(block_num):
        for j in range(block_num):
            y, x = i * block_len, j * block_len
            res[y : y + block_len, x : x + block_len] = blocks[i, j]

    return res

def print_blocks(blocks):
    shape = blocks.shape
    block_num = shape[0]
    block_side = shape[2]

    for i in range(block_num * block_side):
        for j in range(block_num * block_side):
            val = blocks[i // block_side, j // block_side, i % block_side, j % block_side]
            print(f'{' ' if val >= 0 else ''}{val:>3}', end=' ')
            if (j + 1) % block_side == 0:
                print("  ", end="")
        print()
        if (i + 1) % block_side == 0:
            print()

def split_to_subbands(mat):
    side = mat.shape[0] // 2
    LL = mat[:side, :side]
    HL = mat[:side, side:]
    LH = mat[side:, :side]
    HH = mat[side:, side:]
    return LL, HL, LH, HH

def combine_subbands(LL, HL, LH, HH):
    side = LL.shape[0]
    result = np.zeros((side * 2, side * 2))
    result[:side, :side] = LL
    result[:side, side:] = HL
    result[side:, :side] = LH
    result[side:, side:] = HH
    return result

def embedd_bit(bit, t, HL, LH, debug=False, pre=''):
    mHL = np.mean(HL)
    mLH = np.mean(LH)
    mf1 = (t - mHL + mLH) / 2
    mf2 = (t - mLH + mHL) / 2

    diffHL = 0
    diffLH = 0

    if bit == 1 and (mHL - mLH) < t:
        diffHL = mf1
        diffLH = -mf1
    elif bit == 0 and (mLH - mHL) < t:
        diffHL = -mf2
        diffLH = mf2

    if debug:
        print(f'{pre}bit = {bit}')
        print(f'{pre}mf1, mf2 = {mf1:.6f}, {mf2:.6f}')
        print(f'{pre}mHL  - mLH  | t  =>  {mHL:.6f} - {mLH:.6f} = {mHL - mLH:.6f} | {t}')
        print(f'{pre}mHLn - mLHn | t  =>  {mHL + diffHL:.6f} - {mLH + diffLH:.6f} = {(mHL + diffHL) - (mLH + diffLH):.6f} | {t}')
        print(f'{pre}diffHL, diffLH = {diffHL:.4f}, {diffLH:.4f}')
        print()

    return (HL + diffHL, diffHL), (LH + diffLH, diffLH)

def extract_bit(t, HL, LH, diffs, use_new_bit_extraction=False, debug=False, pre=''):
    mHL = np.mean(HL)
    mLH = np.mean(LH)

    if use_new_bit_extraction:
        diff = mHL - mLH
        distT = abs(t - diff)
        distTn = abs(-t - diff)

        if diff >= t or (diff > 0 and distT < distTn):
            bit = 1
        else:
            bit = 0
    else:
        if (mHL - mLH) > t or ((t - (mHL - mLH)) > 0 and (t - (mHL - mLH)) < 1e-6):
            bit = 1
        else:
            bit = 0

    if debug:
        print(f'{pre}mHL - mLH | t => {mHL - mLH} | {t}')
        print(f'{pre}extracted bit: {bit}\n')

    nHL = HL - diffs[0]
    nLH = LH - diffs[1]

    return nHL, nLH, bit

def changed_map2list(changed_map, blocks_num):
    if not isinstance(changed_map, dict):
        raise ValueError('changed_map is not a list, cannot compress')

    keys = np.zeros(blocks_num**2, dtype=bool)
    for key in changed_map:
        keys[key[0] * blocks_num + key[1]] = True
    values = np.array(list(changed_map.values()))
    return (keys, values)

def list2changed_map(keys, values, blocks_num):
    changed_map = {}
    current = 0
    for i in range(len(keys)):
        if keys[i]:
            changed_map[(i // blocks_num, i % blocks_num)] = values[current]
            current += 1
    return changed_map

def compress(keys):
    binary_string = ''.join('1' if key else '0' for key in keys)
    compressed = zlib.compress(binary_string.encode('utf-8'), level=9)
    return compressed

def decompress(compressed):
    decompressed = zlib.decompress(compressed).decode('utf-8')
    keys = np.array([int(char) for char in decompressed])
    return keys

def embedd(img, str, block_size, t, process_params=ProcessParams(), debug=False):
    blocks = split_to_blocks(img, block_size)
    blocks_num = blocks.shape[0]
    
    if debug:
        print('--- EMBEDDING ---')
        print(f'blocks: {blocks_num ** 2}')
        print(f'chars: {int(blocks_num ** 2 / 8)}')
        print('blocks:')
        print_blocks(blocks)

    result_blocks = copy.deepcopy(blocks)

    bit_pos = 0
    changed_map = {}
    try:
        for i in range(blocks_num):
            for j in range(blocks_num):
                if bit_pos == len(str):
                    raise Exception('all bits were embedded before end of img was reached')
                    
                if debug:
                    print(f'for block[{i}, {j}]:')
                    ut.pmat(blocks[i, j], pre='\t')
                
                transformed_block = tr.apply_slt(blocks[i, j])
                if debug:
                    print('\ttransformed:')
                    ut.pmat(transformed_block, pre='\t')

                ll, hl, lh, hh = split_to_subbands(transformed_block)
                (hln, diff_hln), (lhn, diff_lhn) = embedd_bit(int(str[bit_pos]), t, hl, lh, debug=debug, pre='\t')
                bit_pos += 1
                if diff_hln != 0 or diff_lhn != 0:
                    changed_map[(i, j)] = (diff_hln, diff_lhn)

                embedded_block = combine_subbands(ll, hln, lhn, hh)
                if debug:
                    print('\tembedded')
                    ut.pmat(embedded_block, pre='\t')
                    print('\tembedded_block - transformed')
                    ut.pmat(embedded_block - transformed_block, pre='\t')

                result_block_float = tr.apply_inverse_slt(embedded_block)
                result_blocks[i, j] = np.round(result_block_float)
                if debug:
                    print('\tafter inverse slt')
                    ut.pmat(result_block_float, pre='\t')
                    print('\tresult - block')
                    ut.pmat(result_block_float - blocks[i, j], pre='\t')
                    print('\tresult block rounded')
                    ut.pmat(result_blocks[i, j], pre='\t')
                    print('\tresult - block')
                    ut.pmat(result_blocks[i, j] - blocks[i, j], pre='\t')
    
    except Exception as e:
        if debug:
            print(e)

    result_over = combine_blocks(result_blocks).astype(np.int16)
    if debug:
        print('result with over')
        ut.pmat(result_over)

    result = np.clip(result_over, 0, 255)
    over = copy.deepcopy(result_over)
    over = over - result

    if process_params.use_new_side_info:
        changed_map = changed_map2list(changed_map, blocks_num)

    if process_params.compress_side_info:
        keys, values = changed_map
        keys_compressed = compress(keys)
        changed_map = (keys_compressed, values)

    if debug:
        print('result')
        ut.pmat(result)
        print('over')
        ut.pmat(over)
        print(f'changed during embedding: {len(changed_map)}')
        print()

    return result, over, changed_map

def extract(orig, over, changed_map, block_size, t, process_params=ProcessParams(), debug=False):
    img = orig + over

    if debug:
        print('--- EXTRACTING ---')
        print('img with over')
        ut.pmat(img)
    
    blocks = split_to_blocks(img, block_size)
    blocks_num = blocks.shape[0]

    if process_params.compress_side_info:
        keys_compressed, values = changed_map
        keys = decompress(keys_compressed)
        changed_map = (keys, values)

    if process_params.use_new_side_info:
        keys, values = changed_map
        changed_map = list2changed_map(keys, values, blocks_num)
    
    if debug:
        print(f'blocks: {blocks_num ** 2}')
        print(f'chars: {int(blocks_num ** 2 / 8)}')
        print('blocks:')
        print_blocks(blocks)

    result_blocks = copy.deepcopy(blocks)
    mess = ''

    for i in range(blocks_num):
        for j in range(blocks_num):
            if debug:
                print(f'for block[{i}, {j}]:')

            transformed_block = tr.apply_slt(blocks[i, j])
            if debug:
                print('\ttransformed:')
                ut.pmat(transformed_block, pre='\t')

            ll, hl, lh, hh = split_to_subbands(transformed_block)
            hln, lhn, bit = extract_bit(t, hl, lh, changed_map.get((i, j), (0, 0)), use_new_bit_extraction=process_params.use_new_bit_extraction, debug=debug, pre='\t')
            mess = mess + str(bit)
            extracted_block = combine_subbands(ll, hln, lhn, hh)
            if debug:
                print('\textracted_block')
                ut.pmat(extracted_block, pre='\t')
            result_block = tr.apply_inverse_slt(extracted_block)
            result_blocks[i, j] = result_block

    result = combine_blocks(result_blocks)

    if debug:
        print(f"mess: '{mess}'\n")
        print('restored result')
        ut.pmat(result)
        print()
    
    return result, mess

def yellow(val):
    return colored(str(val), 'yellow')

class ProcessStats:
    def __init__(self, orig, embedded, restored, changed_map, mess, mess_restored, t, block_size, process_params=ProcessParams()):
        self.changed = len(changed_map)
        self.pnsr = ut.psnr(orig, embedded)
        self.mat_max_diff = ut.mat_max_diff(orig, restored)
        self.restored_img = (self.mat_max_diff <= 1)
        self.restored_mess = (mess == mess_restored)
        self.restored_count = sum(c1 == c2 for c1, c2 in zip(mess, mess_restored))
        self.restored_percent = self.restored_count / len(mess) * 100
        self.mess_len = len(mess)
        self.t = t
        self.block_size = block_size
        self.image_size = orig.shape[0]
        self.total_cap = int(orig.shape[0] / block_size) ** 2
        self.side_info_size = self.calc_side_info_size(changed_map, process_params)
        self.bits_overcap = self.side_info_size / self.mess_len
        self.time_embedding = 0
        self.time_extracting = 0
        self.process_params = process_params
        self.error_function = self.calc_error()
    
    def calc_side_info_size(self, changed_map, process_params):
        if process_params.use_new_side_info:
            keys, values = changed_map
            if process_params.compress_side_info:
                return len(keys) + len(values) * 64 * 2
            else:
                return (len(keys) // 8) + len(values) * 64 * 2
        else:
            return len(changed_map) * (8 * 2 + 64 * 2)

    def calc_error(self):
        if not self.process_params.use_new_error_function:
            restored = (self.mess_len - self.restored_count) / self.mess_len
            pnsr = 20 / (self.pnsr + 0.00001)
            error = restored * 1 + pnsr * 1
        else:
            restored = (self.mess_len - self.restored_count) / self.mess_len
            pnsr = 20 / (self.pnsr + 0.00001)
            overcap = (self.side_info_size / self.mess_len) / 100
            error = restored * 3 + pnsr * 1 + overcap * 2
        return error
    
    def _to_dict(self):
        return {
            'changed': self.changed,
            'pnsr': self.pnsr,
            'mat_max_diff': float(self.mat_max_diff),
            'restored_img': bool(self.restored_img),
            'restored_mess': self.restored_mess,
            'restored_count': self.restored_count,
            'restored_percent': self.restored_percent,
            'mess_len': self.mess_len,
            't': self.t,
            'block_size': self.block_size,
            'image_size': self.image_size,
            'total_cap': self.total_cap,
            'side_info_size': self.side_info_size,
            'bits_overcap': self.bits_overcap,
            'time_embedding': self.time_embedding,
            'time_extracting': self.time_extracting,
            'error_function': self.error_function,
        }

    def __repr__(self):
        s = ''
        s = s + f't               : {self.t:.6f}\n'
        s = s + f'block size      : {self.block_size}\n'
        s = s + f'changed         : {self.changed}\n'
        s = s + f'PNSR            : {self.pnsr:.2f}\n'
        s = s + f'error function  : {self.error_function}\n'
        s = s + '\n'
        s = s + f'image size      : {self.image_size}x{self.image_size} = {self.image_size ** 2}\n'
        s = s + f'message len     : {self.mess_len} bits\n'
        s = s + f'total image cap : {self.total_cap} bits (blocks)\n'
        s = s + f'side info size  : {self.side_info_size} bits (int8, float64)\n'
        s = s + f'bits overcap    : {self.bits_overcap:.2f}x\n'
        s = s + '\n'
        s = s + f'restored_img    : {ut.pbool(self.restored_img)}\n'
        s = s + f'restored_mess   : {ut.pbool(self.restored_mess)} ({self.restored_count}/{self.mess_len} = {self.restored_percent:.2f} %)\n'
        s = s + f'time embedding  : {self.time_embedding:.3f}s\n'
        s = s + f'time extracting : {self.time_extracting:.3f}s\n'
        s = s + f'time total      : {self.time_embedding + self.time_extracting:.3f}s\n'
        return s

def process(orig, mess, t, block_size, params=ProcessParams()):
    emb_start_time = time.time()
    embedded, over, changed_map = embedd(orig, mess, block_size, t, process_params=params, debug=params.debug)
    emb_end_time = time.time()
    
    ext_start_time = time.time()
    restored, mess_restored = extract(embedded, over, changed_map, block_size, t, process_params=params, debug=params.debug)
    ext_end_time = time.time()

    stats = ProcessStats(orig, embedded, restored, changed_map, mess, mess_restored, t, block_size, process_params=params)
    stats.time_embedding = emb_end_time - emb_start_time
    stats.time_extracting = ext_end_time - ext_start_time

    if params.debug:
        print('restored - orig')
        ut.pmat(restored - orig)
    if params.show:
        ut.show_images((orig, 'orig'), (embedded, 'embedded'), (restored, 'restored'))
    if params.debug:
        print(stats)
    
    return stats
