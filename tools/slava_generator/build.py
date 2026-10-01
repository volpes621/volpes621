"""Build the Slava-class cruiser GLB.

usage:
    python build.py                       # Moskva (121), default output slava_class_cruiser.glb
    python build.py out.glb --ship varyag
    python build.py out.glb --ship ustinov --no-ao

Ships:  moskva  -> pennant 121, name МОСКВА, red stars on the P-1000/P-500 container covers
        ustinov -> pennant 055, name МАРШАЛ УСТИНОВ
        varyag  -> pennant 011, name ВАРЯГ
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ship  # noqa: E402

SHIPS = {
    'moskva': dict(pennant='121', name='МОСКВА', stars=True),
    'ustinov': dict(pennant='055', name='МАРШАЛ УСТИНОВ', stars=False),
    'varyag': dict(pennant='011', name='ВАРЯГ', stars=False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?', default='slava_class_cruiser.glb')
    ap.add_argument('--ship', choices=sorted(SHIPS), default='moskva')
    ap.add_argument('--pennant', help='override the hull number')
    ap.add_argument('--no-ao', action='store_true', help='skip the vertex ambient-occlusion bake')
    a = ap.parse_args()
    cfg = SHIPS[a.ship]
    ship.PENNANT = a.pennant or cfg['pennant']
    ship.SHIP_NAME = cfg['name']
    ship.STARS_ON_CAPS = cfg['stars']
    if a.no_ao:
        ship.AO = None
    ship.build_all(a.out)
    print('written', a.out)


if __name__ == '__main__':
    main()
