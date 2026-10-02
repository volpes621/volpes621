"""Build the Admiral Gorshkov-class (Project 22350) frigate GLB.

usage:
    python build.py                          # Admiral Gorshkov (454), default output admiral_gorshkov_class_frigate.glb
    python build.py out.glb --ship kasatonov
    python build.py out.glb --pennant 417 --no-ao

Ships:  gorshkov  -> pennant 454 (417 until 2020), name АДМИРАЛ ФЛОТА СОВЕТСКОГО СОЮЗА ГОРШКОВ
        kasatonov -> pennant 461, name АДМИРАЛ ФЛОТА КАСАТОНОВ
        golovko   -> pennant 456, name АДМИРАЛ ГОЛОВКО

The generic modules (meshkit, texkit, ao) are shared with ../slava_generator.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, '..', 'slava_generator'))

import ship  # noqa: E402

SHIPS = {
    'gorshkov': dict(pennant='454', name='АДМИРАЛ ФЛОТА СОВЕТСКОГО СОЮЗА ГОРШКОВ'),
    'kasatonov': dict(pennant='461', name='АДМИРАЛ ФЛОТА КАСАТОНОВ'),
    'golovko': dict(pennant='456', name='АДМИРАЛ ГОЛОВКО'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?', default='admiral_gorshkov_class_frigate.glb')
    ap.add_argument('--ship', choices=sorted(SHIPS), default='gorshkov')
    ap.add_argument('--pennant', help='override the hull number')
    ap.add_argument('--no-ao', action='store_true', help='skip the vertex ambient-occlusion bake')
    a = ap.parse_args()
    cfg = SHIPS[a.ship]
    ship.PENNANT = a.pennant or cfg['pennant']
    ship.SHIP_NAME = cfg['name']
    if a.no_ao:
        ship.AO = None
    ship.build_all(a.out)
    print('written', a.out)


if __name__ == '__main__':
    main()
