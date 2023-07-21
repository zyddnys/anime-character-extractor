
import os
import re
import requests
import tqdm
import sys
import hashlib
import cv2
import numpy as np

def get_digest(file_path: str) -> str:
    h = hashlib.sha256()
    BUF_SIZE = 65536 

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(BUF_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def get_filename_from_url(url: str, default: str = '') -> str:
    m = re.search(r'/([^/?]+)[^/]*$', url)
    if m:
        return m.group(1)
    return default

def is_url(s: str):
    return re.search(r'^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$', s) and True

def download_url_with_progressbar(url: str, path: str):
    if os.path.basename(path) in ('.', '') or os.path.isdir(path):
        new_filename = get_filename_from_url(url)
        if not new_filename:
            raise Exception('Could not determine filename')
        path = os.path.join(path, new_filename)

    headers = {}
    downloaded_size = 0
    if os.path.isfile(path):
        downloaded_size = os.path.getsize(path)
        headers['Range'] = 'bytes=%d-' % downloaded_size
        headers['Accept-Encoding'] = 'deflate'

    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if downloaded_size and r.headers.get('Accept-Ranges') != 'bytes':
        print('Error: Webserver does not support partial downloads. Restarting from the beginning.')
        r = requests.get(url, stream=True, allow_redirects=True)
        downloaded_size = 0
    total = int(r.headers.get('content-length', 0))
    chunk_size = 1024

    if r.ok:
        with tqdm.tqdm(
            desc=os.path.basename(path),
            initial=downloaded_size,
            total=total+downloaded_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            with open(path, 'ab' if downloaded_size else 'wb') as f:
                is_tty = sys.stdout.isatty()
                downloaded_chunks = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)

                    # Fallback for non TTYs so output still shown
                    downloaded_chunks += 1
                    if not is_tty and downloaded_chunks % 1000 == 0:
                        print(bar)
    else:
        raise Exception(f'Couldn\'t resolve url: "{url}" (Error: {r.status_code})')
    
def download_model_file(filename: str, url: str, sha256: str) :
    if not os.path.exists(filename) :
        download_url_with_progressbar(url, filename)
    calc_sha256 = get_digest(filename)
    if calc_sha256 != sha256 :
        print(f'Hash mismatch for {filename}, calculated {calc_sha256} != expected {sha256}')
        raise Exception()


def resize_keep_aspect(img: np.ndarray, size: int) :
    ratio = (float(size)/min(img.shape[0], img.shape[1]))
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR), ratio

def resize_keep_aspect_max(img: np.ndarray, size: int):
    ratio = size / max(img.shape[0], img.shape[1])
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    img2 = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
    return img2
