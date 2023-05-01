import torch
import glob
import os
import argparse
from depth_net import load_main_dpt_model
import cv2
from imutils import FileVideoStream
import numpy as np
import time
import PIL.Image as pil
import argparse
import shutil

def prediction(device, model, image, inputSize, targetSize, optimize, useData=True, scale=True):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    #* Simple prediction, add more later
    sample = torch.from_numpy(image).to(device)
    if optimize and torch.device('cuda')==device:
        sample=sample.to(memory_format=torch.channels_last)
        sample=sample.half()
    prediction = model(sample)
    result = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=targetSize[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
    if scale:
        _, result = disp_to_depth(result, 0.1, 100)
    prediction = (result)
    return prediction

# def simple_depth_output(model, image, device):
    # sample = torch.from_numpy(image)

def read_image(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0
    return img

def get_depth_img(depth):
    imgDepth = depth.astype(np.uint16)
    img = pil.fromarray(imgDepth)
    return img

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

class MidasDepth:
    # Predict one->many and many->many
    # NOTE: the depth output, for this application scale it from 0->1 rather than 0->255 for visualization
    def __init__(self, modelPath, scale, saturationDepth) -> None:
        self.midas, self.transform, self.netH, self.netW = load_main_dpt_model(modelPath)
        self.device = torch.device('cuda')
        self.optimize = False
        self.saturationDepth = saturationDepth
        self.scale=scale

    @staticmethod
    def make_output_folder(outputPath):
        imgFullFolder = os.path.join(outputPath, 'color')
        mskFullFolder = os.path.join(outputPath, 'depth')
        if not os.path.isdir(outputPath):
            os.makedirs(imgFullFolder)
            os.makedirs(mskFullFolder)
        return imgFullFolder, mskFullFolder
     
    def predict_folder(self, inputPath, outputPath, nImage=1000):
        # from folder to folder
        self.make_output_folder(outputPath)
        imgNames = os.listdir(inputPath)
        nImgFol = len(imgNames)
        if nImgFol<=nImage:
            nImage = nImgFol
        with torch.no_grad():
            for imgName in imgNames[:nImgFol]:
                imgBaseName = imgName[:-4]
                imgInputPath = os.path.join(inputPath, imgName)
                originalImage = read_image(imgInputPath)
                image = self.transform({'image': originalImage/255})['image']
                imgFullFol, mskFullFol = self.make_output_folder(outputPath)
                imgResPath = os.path.join(imgFullFol, imgName)
                shutil.copy(imgInputPath, imgResPath)
                mskResPath = os.path.join(mskFullFol, imgName)

                # Temporary saving the same as EndoDepth: grayscale with 16 bit
                self.predict_and_save(image, mskResPath)


    def predict_and_save(self, img, dphSavePath):
        result = prediction(device=self.device, model=self.midas, image=img,
                            inputSize=(self.netW, self.netH), targetSize=img[1::-1], optimize=self.optimize, useData=False)
        # result = result*self.scale
        # depth[depth]
        depthImg = get_depth_img(result)
        depthImg.save(dphSavePath)


    def predict_video(self, inputPath, outputPath, nFrames=3000):
        print('Get input from video file')
        assert nFrames%30==0, 'Frame number extract must be diviable by 30'
        fileName = os.path.basename(inputPath)
        baseName, ext = fileName[:-4], fileName[-4:]
        video = FileVideoStream(inputPath).start()
        fps = video.stream.get(cv2.CAP_PROP_FPS)
        vWidth = int(video.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        vHeight = int(video.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vFrame = int(video.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # depthVideoCapture = cv2.VideoWriter(depthVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vWidth, vHeight), True)
        if vFrame<=nFrames:
            nFrames = vFrame
        with torch.no_grad():
            frame_index = 0
            while frame_index<nFrames:
                frame = video.read()
                if frame is not None:
                    name = f'{baseName}_{frame_index}.jpg'
                    original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                    image = self.transform({"image": original_image_rgb/255})["image"]
                    imgFolder, mskFolder = self.make_output_folder(outputPath)
                    imgFullPath = os.path.join(imgFolder, name)
                    mskFullPath = os.path.join(mskFolder, name)
                    cv2.imwrite(frame, imgFullPath)
                    self.predict_and_save(image, mskFullPath)

                    # original_image_bgr = np.flip(original_image_rgb, 2) if side else None
                    # depthProc = calFrameMean(depth)
                    # depthVideoCapture.write(depthProc)
                    # combinedVideoCapture.write(combined)

                    # cv2.imshow('MiDaS Depth Estimation - Press Escape to close window ', content/255)

                    # if output_path is not None:
                        # filename = os.path.join(output_path, 'Camera' + '-' + model_type + '_' + str(frame_index))
                        # cv2.imwrite(filename + ".png", content)
                    

                    alpha = 0.1
                    if time.time()-time_start > 0:
                        fps = (1 - alpha) * fps + alpha * 1 / (time.time()-time_start)  # exponential moving average
                        time_start = time.time()
                    print(f"\rFPS: {round(fps,2)}", end="")
                    print(f'Processed frame:{frame_index}/{nFrames}')

                    if cv2.waitKey(1) == 27:  # Escape key
                        break

                    frame_index += 1
                else:
                    break


    def __call__(self, inputPath, outputPath):
        if os.path.isdir(inputPath):
            print('Running on folder')
            self.predict_folder(inputPath, outputPath)
        else:
            suffix = inputPath[-4:]
            if suffix in ['.mp4', '.avi']:
                print('Running on video')
                self.predict_video(inputPath, outputPath)
            else:
                raise RuntimeError('Must be a folder of images or a video file')
            
def disp_to_depth(disp, minDepth, maxDepth):
    minDisp = 1/maxDepth
    maxDisp = 1/minDepth
    scaledDisp = minDisp+(maxDisp-minDisp)*disp
    depth=1/scaledDisp    
    return scaledDisp, depth


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )
    parser.add_argument('-n', '--num_images', default=1000, help='Number of images that is extracted by midas')
    
    parser.add_argument(
        '--scale',
        type=int,
        default=52.864,
        help='image depth scaling. For Hamlyn dataset the weighted average baseline is 5.2864. It is multiplied by 10 '
             'because the imposed baseline during training is 0.1',
    )

    parser.add_argument(
        '--saturation_depth',
        type=int,
        default=300,
        help='saturation depth of the estimated depth images. For Hamlyn dataset it is 300 mm by default',
    )

    # parser.add_argument('-t', '--model_type',
    #                     default='dpt_beit_large_512',
    #                     help='Model type: '
    #                          'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
    #                          'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
    #                          'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
    #                          'openvino_midas_v21_small_256'
    #                     )

    # # parser.add_argument('-s', '--side',
    #                     action='store_true',
    #                     help='Output images contain RGB and depth images side by side'
    #                     )

    # parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    # parser.set_defaults(optimize=False)

    # parser.add_argument('--height',
    #                     type=int, default=None,
    #                     help='Preferred height of images feed into the encoder during inference. Note that the '
    #                          'preferred height may differ from the actual height, because an alignment to multiples of '
    #                          '32 takes place. Many models support only the height chosen during training, which is '
    #                          'used automatically if this parameter is not set.'
    #                     )
    # parser.add_argument('--square',
    #                     action='store_true',
    #                     help='Option to resize images to a square resolution by changing their widths when images are '
    #                          'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
    #                          'images is tried to be preserved if supported by the model.'
    #                     )
    # parser.add_argument('--grayscale',
    #                     action='store_true',
    #                     help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
    #                          'which is used by default, is better for visibility, it does not allow storing 16-bit '
    #                          'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
    #                          'colormap.'
    #                     )

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_parser()
    
    depthPred = MidasDepth(args.model_weights, args.scale, args.saturation_depth)
    inputPath = args.input_path
    outputPath = args.output_path
    depthPred(inputPath, outputPath)
