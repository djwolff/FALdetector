import argparse
import mimetypes
import os
import sys
import imageio
import torch
from PIL import Image
import torchvision.transforms as transforms
import tqdm

from networks.drn_seg import DRNSub
from utils.tools import *
from utils.visualize import *

def get_file_type(path):
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime

def load_classifier(model_path, gpu_id):
    if torch.cuda.is_available() and gpu_id != -1:
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    model = DRNSub(1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.device = device
    model.eval()
    return model


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

def classify_fake_image(model, img_path, no_crop=False,
                  model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    # Data preprocessing
    im_w, im_h = Image.open(img_path).size
    if no_crop:
        face = Image.open(img_path).convert('RGB')
    else:
        faces = face_detection(img_path, verbose=False, model_file=model_file)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(model.device)

    # Prediction
    with torch.no_grad():
        prob = model(face_tens.unsqueeze(0))[0].sigmoid().cpu().item()
    return prob

def classify_fake_video(model, vid_path, no_crop=False,
                  model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    
    reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(vid_path)
    meta = reader.get_meta_data()
    _ = meta['size']
    read_iter = reader.iter_data()
    nframes = reader.count_frames()

    # loading bar:
    bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1)

    totp, minp, maxp = 0, 1, 0
    for i, frame in enumerate(read_iter):
        # Data preprocessing
        if no_crop:
            face = Image.fromarray(frame).convert('RGB')
        else:
            faces = face_detection_image(Image.fromarray(frame), verbose=False, model_file=model_file)
            if len(faces) == 0:
                nframes -= 1
                continue
            face, box = faces[0]
        face = resize_shorter_side(face, 400)[0]
        face_tens = tf(face).to(model.device)

        # Prediction
        with torch.no_grad():
            prob = model(face_tens.unsqueeze(0))[0].sigmoid().cpu().item()
        totp += prob
        minp, maxp = min(minp, prob), max(maxp, prob)
        bar.update()

    return totp / nframes, minp, maxp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--model_path", required=True, help="path to the drn model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()
    model = load_classifier(args.model_path, args.gpu_id)

    # TODO: process a video
    # Have an average, min, and max probilities for each frame of the video
    input_path = args.input_path
    filetype = get_file_type(input_path)
    if filetype == 'video':
        avg, min, max = classify_fake_video(model, input_path, args.no_crop)
        print("Probablities being modified by Photoshop FAL:")
        print("Average probability of being modified: {:.2f}%".format(avg*100))
        print("Minimum probability of being modified: {:.2f}%".format(min*100))
        print("Maximum probability of being modified: {:.2f}%".format(max*100))
    elif filetype == 'image':
        prob = classify_fake_image(model, input_path, args.no_crop)
        print("Probibility being modified by Photoshop FAL: {:.2f}%".format(prob*100))
    else:
        print(f'Can\'t process file {input_path}')

