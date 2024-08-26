from pathlib import Path
import shutil

import cv2

standard_w = 256
data_home = Path(__file__).parent
resized_dir = data_home.joinpath('resized')
shutil.rmtree(resized_dir)
resized_dir.mkdir()

raw_files = [f for f in data_home.joinpath('i').iterdir() if f.is_file()]
for raw_path in raw_files:
    if (raw_img := cv2.imread(str(raw_path))) is None:
        raise RuntimeError(f'Cannot read {raw_path}')

    resized_path = resized_dir.joinpath(raw_path.name)
    if raw_img.shape[1] < standard_w:
        resize_rate = standard_w / raw_img.shape[1]
        resized_h = int(raw_img.shape[0] * resize_rate)
        resized_img = cv2.resize(raw_img, (standard_w, resized_h))
        if not cv2.imwrite(str(resized_path), resized_img):
            raise RuntimeError(f'Cannot create {resized_path}')
    else:
        if not cv2.imwrite(str(resized_path), raw_img):
            raise RuntimeError(f'Cannot create {resized_path}')

assert len(raw_files) == len(list(resized_dir.iterdir())), 'Not all images are resized'
