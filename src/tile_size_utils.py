from functools import reduce
from itertools import product
from typing import Tuple

# Chosen from: https://www.digitalcitizen.life/what-screen-resolution-or-aspect-ratio-what-do-720p-1080i-1080p-mean
STANDARD_RESOLUTIONS = dict()
for height in (360, 720, 1080, 1440, 2160, 4320):
    STANDARD_RESOLUTIONS[f'{height}p'] = (height * 16 // 9, height)
STANDARD_RESOLUTIONS['HD'] = STANDARD_RESOLUTIONS['720p']
STANDARD_RESOLUTIONS['FHD'] = STANDARD_RESOLUTIONS['1080p']
STANDARD_RESOLUTIONS['QHD'] = STANDARD_RESOLUTIONS['1440p']
STANDARD_RESOLUTIONS['4K'] = STANDARD_RESOLUTIONS['2160p']
STANDARD_RESOLUTIONS['8K'] = STANDARD_RESOLUTIONS['4320p']

def to_literal(tile_size: Tuple[int]):
    w, h = tile_size
    return f'{w}x{h}'


def from_literal(size_literal: str):
    if size_literal.count('x') == 1:
        w, h = size_literal.split('x')
        return (int(w), int(h))
    if size_literal[-1] == 'p':
        h = int(size_literal[:-1])
        return (16 * h // 9, h)
    if size_literal in STANDARD_RESOLUTIONS:
        return STANDARD_RESOLUTIONS.get(size_literal)
    return None

SEMI_STANDARD_RESOLUTIONS = [
    # 4:3 aspect ratios
    (640, 480), (800, 600), (960, 720),
    (1024, 768), (1280, 960), (1400, 1050),
    (1440, 1080), (1600, 1200), (1856, 1392),
    (1920, 1440), (2048, 1536),
]
SEMI_STANDARD_RESOLUTIONS = {to_literal(size): size for size in SEMI_STANDARD_RESOLUTIONS}

OTHER_RESOLUTIONS = [
    # 16:10 aspect ratios
    (1280, 800), (1440, 900), (1680, 1050),
    (1920, 1200), (2560, 1600)
]
OTHER_RESOLUTIONS = {to_literal(size): size for size in OTHER_RESOLUTIONS}
for height in (240, 480):
    OTHER_RESOLUTIONS[f'{height}p'] = (height * 16 // 9, height)

def get_divisors(n, ub=None):
    d = []
    if ub is None:
        ub = n
    for i in range(1, n // 2 + 1):
        if i > ub:
            break
        if n % i == 0:
            d.append(i)
    if ub >= n:
        d.append(n)
    return d


def gcd(*args):
    def gcd2(a, b):
        r = a % b
        if r == 0:
            return b
        else:
            return gcd2(b, r)

    return reduce(gcd2, args)


def lcm(*args):
    def lcm2(a, b):
        return a * b // gcd(a, b)

    return reduce(lcm2, args)


def max_power_of_2(num: int):
    i = 0
    while True:
        if num % (2 ** (i + 1)) != 0:
            return i
        i += 1


def block_dimension(resolution, tile_size):
    W, H = resolution
    w, h = tile_size
    assert W % w == 0 and H % h == 0, f'Tile size {tile_size} is not suitable for {resolution}'
    return (W // w, H // h)


def calc_score(tile_size, batches_per_step, num_steps, num_ipus):
    buffer_rate = (tile_size[0] + 4) * (tile_size[1] + 4) / 15396
    return num_ipus / (0.036 * (num_steps - 1) + 1.72 * batches_per_step * num_steps * buffer_rate)


def parse_resolution(res_literal: str):
    res = from_literal(res_literal)
    if res_literal not in STANDARD_RESOLUTIONS and res is None:
        raise Exception(f'{res_literal} is not a valid resolution')
    if res is None:
        res = STANDARD_RESOLUTIONS.get(res_literal)
    return res


def get_tile_size_candidates(resolutions, min_size, max_size):
    widths = [res[0] for res in resolutions]
    heights = [res[1] for res in resolutions]
    tile_widths = get_divisors(gcd(*widths))
    tile_heights = get_divisors(gcd(*heights))
    if min_size:
        tile_widths = list(filter(lambda x: x >= min_size[0], tile_widths))
        tile_heights = list(filter(lambda x: x >= min_size[1], tile_heights))
    if max_size:
        tile_widths = list(filter(lambda x: x <= max_size[0], tile_widths))
        tile_heights = list(filter(lambda x: x <= max_size[1], tile_heights))

    return list(product(tile_widths, tile_heights))


def edsr_tile_sizes(res_literals, min_size, max_size, best_only):
    resolutions = list(map(parse_resolution, res_literals))
    tile_size_candidates = get_tile_size_candidates(resolutions, min_size, max_size)

    infos = dict()
    for tile_size in tile_size_candidates:
        block_dims = [block_dimension(res, tile_size) for res in resolutions]
        num_blocks = [reduce(lambda x, y: x * y, block_dim) for block_dim in block_dims]
        bs_candidates = get_divisors(gcd(*num_blocks))
        block_dims = dict(zip(res_literals, block_dims))
        info = {
            'block_dims': block_dims,
            'options': []
        }
        for batches_per_step in bs_candidates:
            option = {'batches_per_step': batches_per_step, 'config': dict()}
            for res_literal, block_dim in block_dims.items():
                num_block = block_dim[0] * block_dim[1]
                total_loads = num_block // batches_per_step
                num_ipus = max(get_divisors(total_loads, ub=16))
                num_steps = total_loads // num_ipus
                score = calc_score(tile_size, batches_per_step, num_steps, num_ipus)
                option['config'][res_literal] = {
                    'num_ipus': num_ipus,
                    'num_steps': num_steps,
                    'score': score
                }
            info['options'].append(option)
        if best_only:
            options = info['options']
            argmax = 0
            maxval = -1
            for idx, option in enumerate(options):
                throughputs = [opt['score'] for opt in option['config'].values()]
                min_throughput = min(*throughputs) if len(throughputs) > 1 else throughputs[0]
                if min_throughput > maxval:
                    maxval = min_throughput
                    argmax = idx
            info['options'] = options[argmax]
        infos[to_literal(tile_size)] = info
    if best_only:
        best_throughput = -1
        min_throughputs = dict()
        for size_literal, info in infos.items():
            config = info['options']['config']
            throughputs = [opt['score'] for opt in config.values()]
            min_throughput = min(*throughputs) if len(throughputs) > 1 else throughputs[0]
            min_throughputs[size_literal] = min_throughput
            if min_throughput > best_throughput:
                best_throughput = min_throughput
        best_tiles = [size_literal for size_literal, min_throughput in min_throughputs.items() if
                      min_throughput == best_throughput]

        infos = {size_literal: infos.get(size_literal) for size_literal in best_tiles}

    return infos


def mdsr_tile_sizes(res_literals, include_invalid=False):
    resolutions = list(map(parse_resolution, res_literals))
    for res_literal, (w, h) in zip(res_literals, resolutions):
        import warnings
        if w % 40 != 0:
            warnings.warn(f'{res_literal}: width {w} is not a multiple of 40')
        if h % 40 != 0:
            warnings.warn(f'{res_literal}: width {h} is not a multiple of 40')
    tile_sizes = [(gcd(*res), gcd(*res)) for res in resolutions]
    block_dims = [(w // size[0], h // size[0]) for (w, h), size in zip(resolutions, tile_sizes)]
    num_blocks = [dim[0] * dim[1] for dim in block_dims]
    for idx, (size, nblocks) in enumerate(zip(tile_sizes, num_blocks)):
        assert (size[0] * nblocks) % 16 == 0, f'You may consider removing {res_literals[idx]}'
        max_power = max_power_of_2(nblocks)
        if max_power >= 4:
            continue
        extra_power = 4 - max_power
        scale_x = 2 ** (extra_power // 2)
        scale_y = (2 ** extra_power) // scale_x
        block_dims[idx] = (block_dims[idx][0] * scale_x, block_dims[idx][1] * scale_y)
        tile_sizes[idx] = (size[0] // scale_x, size[1] // scale_y)
        num_blocks[idx] = block_dims[idx][0] * block_dims[idx][1]

    batches_per_step = gcd(*[nblocks // 16 for nblocks in num_blocks])
    num_steps = [nblocks // 16 // batches_per_step for nblocks in num_blocks]

    sr_tile_size = lcm(*[size[0] for size in tile_sizes])
    info = {
        'batches_per_step': batches_per_step,
        'num_ipus': 16,
        'output_tile_size': (sr_tile_size, sr_tile_size)
    }
    for res_literal, tile_size, block_dim, num_step in zip(res_literals, tile_sizes, block_dims, num_steps):
        scale = sr_tile_size // tile_size[0]
        config = {
            'tile_size': tile_size,
            'block_dim': block_dim,
            'num_steps': num_step,
            'scale': scale
        }
        if scale not in (2, 3):
            import json
            import warnings
            msg = f'You should consider replacing input resolution {res_literal} by something else. ' \
                + f'Otherwise you may choose to use an unreasonable config:\n{json.dumps(config)}'
            warnings.warn(msg)
            if include_invalid:
                info[res_literal] = config
        info[res_literal] = config

    return info
