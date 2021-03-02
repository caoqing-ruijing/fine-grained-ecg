# -*- coding: utf-8 -*-

import os
import glob
import pdf2image
from PIL import Image
import numpy as np
import cv2

Image.MAX_IMAGE_PIXELS = 2300000000
PDF_CONFIG = {'DPI': 200, 'FORMAT': 'ppm', 'USE_CROPBOX': True, 'STRICT': True}


def pdftopil(each_filename, pdf_config=None):
    if pdf_config is None:
        pdf_config = PDF_CONFIG
    # output_folder=OUTPUT_FOLDER,
    pil_images = pdf2image.convert_from_path(each_filename, dpi=pdf_config["DPI"], fmt=pdf_config["FORMAT"],
                                             thread_count=1, use_cropbox=pdf_config["USE_CROPBOX"],
                                             strict=pdf_config["STRICT"], grayscale=True)
    return pil_images


def save_images(pil_images, output_name):
    # This method helps in converting the images in PIL Image file format to the required image format

    for image in pil_images:
        image.save(output_name)


def crop(image_obj, coords):
    """
    @param image_obj: image in PIL.Image format
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    """
    cropped_image = image_obj.crop(coords)
    return cropped_image


def get_filelist(root):
    Filelist = []
    for home, dirs, files in os.walk(root):
        for filename in files:
            if filename.endswith('.pdf'):
                Filelist.append(os.path.join(home, filename))
    return Filelist


def remove_symbol(dilated):

    default_pixel = 0

    dilated[80:108, 65:83] = default_pixel
    dilated[350:382, 65:100] = default_pixel
    dilated[635:663, 65:98] = default_pixel
    dilated[908:935, 65:93] = default_pixel

    dilated[84:110, 558:611] = default_pixel
    dilated[361:385, 558:611] = default_pixel
    dilated[635:664, 558:611] = default_pixel

    dilated[86:112, 1050:1084] = default_pixel
    dilated[361:385, 1050:1084] = default_pixel
    dilated[633:663, 1050:1084] = default_pixel

    dilated[86:112, 1538:1577] = default_pixel
    dilated[361:385, 1538:1577] = default_pixel
    dilated[633:663, 1538:1577] = default_pixel

    dilated[0:5, :] = default_pixel
    dilated[:, 0:5] = default_pixel

    dilated[:, 2083:2088] = default_pixel
    dilated[1106:1111, :] = default_pixel

    dilated[1053:1100, 8:780] = default_pixel
    dilated[1080:1150, 1450:1658] = default_pixel

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

    i = 0
    for istat in stats:
        if istat[4] < 30:
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(dilated, tuple(istat[0:2]), tuple(
                istat[0:2]+istat[2:4]), 0, thickness=-1)  # 26
        i = i+1

    return dilated


def image_denoise(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_raw = cv2.bitwise_not(img)

    th2 = cv2.adaptiveThreshold(
        img_raw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontal = th2
    vertical = th2
    rows, cols = horizontal.shape
    horizontalsize = cols / 5
    horizontalsize = int(horizontalsize)

    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

    verticalsize = cols / 20
    verticalsize = int(verticalsize)
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    bit_xor = cv2.bitwise_xor(img_raw, horizontal)
    bit_xor = cv2.bitwise_xor(bit_xor, vertical)

    edges = cv2.Canny(image=bit_xor, threshold1=100, threshold2=950)  # 1000
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    dilated = remove_symbol(edges)
    dilated = cv2.bitwise_not(dilated)

    return dilated


def process_single_pdf(pdf_path, target_dir, crop_bottom=False,
                       need_denoise=False, skip_crop=False,
                       save=True, pdf_config=None):
    try:
        pil_images = pdftopil(pdf_path, pdf_config)
        assert len(pil_images) == 1
        img = pil_images[0]

        if skip_crop:
            assert need_denoise == False, "Not Implemented"
        else:
            # TODO: different crop boxs for different DPIs
            if crop_bottom:
                # without the bottom lead
                img = crop(img, (88, 398, 2178, 1280))
            else:
                img = crop(img, (88, 398, 2178, 1510))  # with the bottom lead

        if need_denoise:
            img = np.array(img)
            img = image_denoise(img)
            img = Image.fromarray(img)

        if save:
            save_path = os.path.join(target_dir, pdf_path.split(
                '/')[-1].replace('.pdf', '.png'))
            img.save(save_path)
        else:
            return img
    except Exception as e:
        print(e)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import multiprocessing
    from tqdm import tqdm
    from functools import partial

    parser = ArgumentParser(
        description="Convert ECG pdf to image in mulitprocess")
    parser.add_argument("--root", type=str,
                        help="the root of input pdfs")
    parser.add_argument("--output", type=str,
                        help="the directory for saving outputs")
    parser.add_argument("--dpi", type=int, default=200,
                        help="the dpi for the output")
    parser.add_argument("--crop-bottom", action="store_true",
                        help="crop the bottom row")
    parser.add_argument("--denoise", action="store_true",
                        help="denoise before saving outputs")
    parser.add_argument("--skip-crop", action="store_true",
                        help="keep the original result without crop")

    args = parser.parse_args()

    # input folder for pdf, pdf can be saved in sub-folder
    root = args.root  # '/Users/ndu/Documents/Projects/Ruijin/Cardiology/data/pdf2'
    # output folder for jpg
    # '/Users/ndu/Documents/Projects/Ruijin/Cardiology/data/pdf2_output/'
    target_dir = args.output
    crop_bottom = args.crop_bottom
    need_denoise = args.denoise
    skip_crop = args.skip_crop
    pdf_config = PDF_CONFIG
    pdf_config["DPI"] = args.dpi

    Filelist = get_filelist(root)
    print(len(Filelist))

    process_pdf_fn = partial(process_single_pdf,
                             target_dir=target_dir,
                             crop_bottom=crop_bottom,
                             need_denoise=need_denoise,
                             skip_crop=skip_crop, 
                             pdf_config=pdf_config)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pool = multiprocessing.Pool(processes=4)
    with tqdm(total=len(Filelist)) as progress_bar:
        for _ in pool.imap_unordered(process_pdf_fn, Filelist):
            progress_bar.update(1)
    print('Done')
