import argparse
import json

from tile_size_utils import edsr_tile_sizes, mdsr_tile_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-size', nargs=2, type=int, default=[50, 50])
    parser.add_argument('--max-size', nargs=2, type=int, default=[250, 250])
    parser.add_argument('--show-all', action='store_true')
    parser.add_argument('--model', type=str, default='EDSR')
    parser.add_argument('-r', '--resolutions', nargs='+', type=str, required=True)
    args = parser.parse_args()
    print(args)
    assert args.model in ('EDSR', 'MDSR')


    if args.model == 'EDSR':
        guide = '[EDSR Tile Sizes]'
        recommendation = edsr_tile_sizes(args.resolutions, args.min_size, args.max_size, not args.show_all)
    else:
        guide = '[MDSR Tile Sizes]'
        recommendation = mdsr_tile_sizes(args.resolutions, include_invalid=True)

    print(f'{args.model} Tile Sizes')
    print(json.dumps(recommendation, indent=2))
