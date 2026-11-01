"""Read YAML settings and launch the Blender renderer."""
import argparse
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]

def build_command(args, settings):
    """Send settings as one JSON argument; Blender does not need PyYAML."""
    settings = dict(settings, model_pattern=args.model_filter, save_scene=args.save_scene)
    return [args.blender, '--background', '--factory-startup', '--python-exit-code', '1',
            '--python', str(ROOT / 'scripts/generate_renders.py'), '--', json.dumps(settings)]


def main():
    parser = argparse.ArgumentParser(description='Render synthetic plankton images using YAML settings.')
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/render_config.yaml')
    parser.add_argument('--save-scene', action='store_true', help='Save the first posed scene for inspection.')
    parser.add_argument('--model-filter', default='*', help='Model filename glob, e.g. Micrasterias*')
    parser.add_argument('--blender', default='D:/Blender 5.2/blender.exe', help='Blender executable path')
    args = parser.parse_args()

    # Load YAML settings
    settings = yaml.safe_load(args.config.open('r'))

    # All YAML input/output paths use the repository root as their base.
    subprocess.run(build_command(args, settings), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
