"""そのディレクトリ内にある画像全てを検知し、余白を追加して画像幅と高さを揃える"""

import os

from PIL import Image

DIRPATH = os.path.dirname(os.path.abspath(__file__))


def triming(pil_img: Image.Image, width: int, height: int, color: "tuple[int, int, int]" = (255, 255, 255)):
    """トリミング・余白を追加する"""
    result = Image.new(pil_img.mode, (width, height), color)
    result.paste(pil_img, (0, 0))
    return result


if __name__ == "__main__":
    max_width = 0
    max_height = 0
    target_image: "list[Image.Image]" = []
    target_filename: "list[str]" = []

    # ディレクトリ内にある全てのファイルを取得する
    files = os.listdir(DIRPATH)
    target_files = [f for f in files if os.path.isfile(os.path.join(DIRPATH, f))]

    # 画像の場合、pillowオブジェクトで開き、最大の幅と高さを取得する
    for file in target_files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".gif"):
            im = Image.open(os.path.join(DIRPATH, file))
            width, height = im.size
            target_image.append(im)
            target_filename.append(file)

            max_width = max(max_width, width)
            max_height = max(max_height, height)

    print(f"triming -> width: {max_width}, height: {max_height}")

    # 余白を追加し上書き保存する
    for image, name in zip(target_image, target_filename):
        image = triming(image, max_width, max_height)
        image.save(os.path.join(DIRPATH, name))
