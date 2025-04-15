import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
from collections import OrderedDict

import detection.craft_utils as craft_utils
import detection.imgproc as imgproc
import detection.file_utils as file_utils
from detection.craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, canvas_size=1280, mag_ratio=1.5, show_time=False):
    t0 = time.time()

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def run_craft_text_detection(
        input_dir,
        model_path='weights/craft_mlt_25k.pth',
        refiner_model_path='weights/craft_refiner_CTW1500.pth',
        use_cuda=True,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=1280,
        mag_ratio=1.5,
        poly=False,
        show_time=False,
        use_refiner=False,
        result_folder='../data/detected/'):

    if not os.path.isdir(result_folder):
        os.makedirs(result_folder, exist_ok=True)

    image_list, _, _ = file_utils.get_files(input_dir)

    net = CRAFT()

    print('Loading weights from checkpoint (' + model_path + ').')
    if use_cuda:
        net.load_state_dict(copyStateDict(torch.load(model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if use_refiner:
        from src.detection.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model_path + ')')
        if use_cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path, map_location='cpu')))

        refine_net.eval()
        poly = True

    t = time.time()

    for k, image_path in enumerate(image_list):
        print("Running craft on Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(
            net, image, text_threshold, link_threshold, low_text, use_cuda,
            poly, refine_net, canvas_size, mag_ratio, show_time
        )

        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        mask_file = result_folder + "/res_" + filename + '_mask.jpg'

        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys)

    print("\nElapsed time : {}s".format(time.time() - t))