import os
import re
import math
import random
import argparse
from PIL import Image, UnidentifiedImageError

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 4K ultrawide wallpaper collage with random or grouped scatter"
    )
    parser.add_argument(
        "--mode", choices=("random", "grouped"), default="random",
        help="Placement mode: 'random' for free scatter, 'grouped' to cluster by filename prefix"
    )
    parser.add_argument(
        "--source_dir", default="Wallpaper Creation",
        help="Folder containing source images"
    )
    parser.add_argument(
        "--output", default="ultrawide_collage.png",
        help="Filename for the generated wallpaper"
    )
    parser.add_argument(
        "--width", type=int, default=5120,
        help="Canvas width in pixels (ultrawide 4K default = 5120)"
    )
    parser.add_argument(
        "--height", type=int, default=2160,
        help="Canvas height in pixels (ultrawide 4K default = 2160)"
    )
    parser.add_argument(
        "--bg_color", default="0,0,0",
        help="Background color as R,G,B (default = '0,0,0')"
    )
    return parser.parse_args()

def get_group_name(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"\s*\d+$", "", name)

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

def load_image(path):
    """Attempt to open an image and convert to RGBA, or return None if it fails."""
    try:
        return Image.open(path).convert("RGBA")
    except (UnidentifiedImageError, OSError) as e:
        print(f"Warning: skipping '{os.path.basename(path)}' ({e})")
        return None

def main():
    args = parse_args()
    src = args.source_dir
    out = args.output
    W, H = args.width, args.height
    bg = tuple(map(int, args.bg_color.split(",")))
    mode = args.mode

    # Collect only files (no directories) with valid extensions
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
    files = [
        f for f in os.listdir(src)
        if os.path.isfile(os.path.join(src, f)) and
           os.path.splitext(f.lower())[1] in valid_ext
    ]
    if not files:
        print(f"No valid image files found in '{src}'.")
        return

    canvas = Image.new("RGBA", (W, H), bg + (255,))

    if mode == "random":
        for fname in files:
            img = load_image(os.path.join(src, fname))
            if img is None:
                continue

            angle = random.uniform(-5, 5)
            rotated = img.rotate(angle, expand=True)
            rw, rh = rotated.size

            if rw > W or rh > H:
                print(f"Skipping '{fname}': rotated size {rw}×{rh} exceeds canvas {W}×{H}")
                continue

            x_max, y_max = W - rw, H - rh
            x = random.randint(0, x_max)
            y = random.randint(0, y_max)

            canvas.paste(rotated, (x, y), rotated)

    else:  # grouped mode
        # Group by filename prefix (strip trailing digits)
        groups = {}
        for fname in files:
            key = get_group_name(fname)
            groups.setdefault(key, []).append(fname)

        for key, fnames in groups.items():
            # Pre-calc total area to size the cluster
            total_area = 0
            dims = []
            for fname in fnames:
                path = os.path.join(src, fname)
                img = load_image(path)
                if img is None:
                    continue
                w, h = img.size
                dims.append((fname, w, h))
                total_area += w * h

            if not dims:
                continue

            cluster_dim = min(math.sqrt(total_area) * 2, W, H)
            cx = random.uniform(cluster_dim/2, W - cluster_dim/2)
            cy = random.uniform(cluster_dim/2, H - cluster_dim/2)

            for fname, w, h in dims:
                path = os.path.join(src, fname)
                img = load_image(path)
                if img is None:
                    continue

                angle = random.uniform(-5, 5)
                rotated = img.rotate(angle, expand=True)
                rw, rh = rotated.size

                if rw > W or rh > H:
                    print(f"Skipping '{fname}': rotated size {rw}×{rh} exceeds canvas")
                    continue

                offset_x = random.uniform(-cluster_dim/2, cluster_dim/2)
                offset_y = random.uniform(-cluster_dim/2, cluster_dim/2)
                x = int(cx + offset_x - rw/2)
                y = int(cy + offset_y - rh/2)

                x = clamp(x, 0, W - rw)
                y = clamp(y, 0, H - rh)

                canvas.paste(rotated, (x, y), rotated)

    # Save as PNG for lossless quality
    canvas.convert("RGB").save(out, format="PNG")
    print(f"Collage saved to '{out}'")

if __name__ == "__main__":
    main()
