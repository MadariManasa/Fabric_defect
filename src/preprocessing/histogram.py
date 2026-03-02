import os
import cv2
import numpy as np


def smooth_histogram(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Return an image whose intensities have been remapped according
    to a smoothed version of its histogram.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (8-bit).
    kernel_size : int
        Size of the moving average kernel used to smooth the histogram.
    """

    # compute histogram
    h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    # smooth by convolving with a 1-D averaging kernel
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    h_smooth = np.convolve(h, kernel, mode='same')

    # build cumulative distribution function (CDF) from smoothed histogram
    cdf = np.cumsum(h_smooth)
    cdf = 255 * (cdf / cdf[-1])        # normalize to 0..255
    lut = cdf.astype('uint8')

    # apply lookup table
    return cv2.LUT(img, lut)


def process_directory(src_dir: str, dst_dir: str, kernel_size: int = 5):
    """Apply histogram smoothing to every image in a directory tree.

    The function walks ``src_dir`` recursively, smoothing each detected image
    and writing the result to the corresponding location under ``dst_dir``.
    Subdirectories are recreated in the destination tree.

    Parameters
    ----------
    src_dir : str
        Path to source images (may contain nested folders).
    dst_dir : str
        Directory where processed images will be saved. Created if needed.
    kernel_size : int
        Smoothing kernel size passed to :func:`smooth_histogram`.
    """
    for root, _, files in os.walk(src_dir):
        # compute relative path from src_dir to current directory
        rel = os.path.relpath(root, src_dir)
        out_root = os.path.join(dst_dir, rel) if rel != '.' else dst_dir
        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            path_in = os.path.join(root, fname)
            path_out = os.path.join(out_root, fname)

            img = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"skipping {path_in} (couldn't read)")
                continue

            smoothed = smooth_histogram(img, kernel_size=kernel_size)
            cv2.imwrite(path_out, smoothed)
            print(f"processed {path_in} -> {path_out}")


if __name__ == '__main__':
    # simple command-line interface with sensible defaults
    import argparse
    from pathlib import Path

    # default paths under the repository's data folder
    # __file__ is .../src/preprocessing/histogram.py; go up three levels to project root
    repo_root = Path(__file__).parent.parent.parent
    default_src = repo_root / 'data' / 'captured'
    default_dst = repo_root / 'data' / 'processed'

    parser = argparse.ArgumentParser(
        description='Histogram smoothing for a folder of images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src', nargs='?', default=str(default_src),
                        help='source directory (raw images); default is data/captured')
    parser.add_argument('dst', nargs='?', default=str(default_dst),
                        help='destination directory for processed images; default is data/processed')
    parser.add_argument('--kernel', type=int, default=5,
                        help='smoothing kernel size')
    args = parser.parse_args()

    # ensure source/destination directories are absolute
    src_path = os.path.abspath(args.src)
    dst_path = os.path.abspath(args.dst)

    # if source doesn't exist, create it and warn the user
    if not os.path.isdir(src_path):
        print(f"source directory '{src_path}' does not exist; creating it and exiting")
        os.makedirs(src_path, exist_ok=True)
        # also create dest so structure is ready
        os.makedirs(dst_path, exist_ok=True)
        print("place your raw images in the source folder and re-run the script")
    else:
        process_directory(src_path, dst_path, kernel_size=args.kernel)
