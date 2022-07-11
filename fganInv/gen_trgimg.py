import os
os.chdir('/home/xsy/idinvert_pytorch-mycode/') # convenient for debug
import argparse
# from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from dataset import brush_stroke_mask
import torch
import cv2
import glob
from PIL import Image, ImageDraw
import imgaug.augmenters as iaa

def _get_tensor_value(tensor):
  return tensor.cpu().detach().numpy()
def postprocess( images):
    """Postprocesses the output images if needed.
    This function assumes the input numpy array is with shape [batch_size,
    channel, height, width]. Here, `channel = 3` for color image and
    `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, height, width, channel].

    NOTE: The channel order of output images will always be `RGB`.

    Args:
      images: The raw outputs from the generator.

    Returns:
      The postprocessed images with dtype `numpy.uint8` and range [0, 255].

    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not
        with shape [batch_size, channel, height, width].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    if images.ndim != 4 or images.shape[1] !=3:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to '
                       f'{3}!\n'
                       f'But {images.shape} is received!')
    images = (images +1) * 255 / (2)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)

    return images
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str,default='styleganinv_ffhq256', help='Name of the GAN model.') #todo
  parser.add_argument('--data_root', type=str,default='/home/xsy/datasets/evaluationt_img/src',
                      help='List of images to invert.')
  # parser.add_argument('--image_list', type=str,default='/home/xsy/FFHQ_256_png',
  #                     help='List of images to invert.')
  parser.add_argument('--output_dir', type=str, default='/home/xsy/datasets/evaluationt_img',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  # parser.add_argument('--learning_rate', type=float, default=0.01,
  #                     help='Learning rate for optimization. (default: 0.01)')
  # parser.add_argument('--num_iterations', type=int, default=100,
  #                     help='Number of optimization iterations. (default: 100)')
  # parser.add_argument('--num_results', type=int, default=5,
  #                     help='Number of intermediate optimization results to '
  #                          'save for each sample. (default: 5)')
  # parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
  #                     help='The perceptual loss scale for optimization. '
  #                          '(default: 5e-5)')
  # parser.add_argument('--loss_weight_enc', type=float, default=2.0,
  #                     help='The encoder loss scale for optimization.'
  #                          '(default: 2.0)')
  # parser.add_argument('--viz_size', type=int, default=256,
  #                     help='Image size for visualization. (default: 256)')
  # parser.add_argument('--gpu_id', type=str, default='0',
  #                     help='Which GPU(s) to use. (default: `0`)')
  # parser.add_argument('--image_size', type=int, default=256,
  #                     help='the image size in training dataset (defaults; 256)')
  # parser.add_argument('--batch_size', type=int, default=1,
  #                     help='the batch size in one picture ')
  return parser.parse_args()

def main():
  """Main function."""
  args = parse_args()
  # assert os.path.exists(args.data_root)
  img_list = sorted(glob.glob(args.data_root + '/*.*'))
  image_list_name = os.path.splitext(os.path.basename(args.data_root))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  os.mkdir(os.path.join(output_dir,'trg-gnsp'))#todo
  # os.mkdir(os.path.join(output_dir,'src'))
  # os.mkdir(os.path.join(output_dir,'trg'))

  file_handle=open(output_dir+'/image_list-gnsp.txt',mode='w') #todo
  file_cont=[]
  # cloud = iaa.CloudLayer(
  #     intensity_mean=(196, 255),
  #     intensity_freq_exponent=(-2.5, -2.0),
  #     intensity_coarse_scale=10,
  #     alpha_min=0,
  #     alpha_multiplier=(0.25, 0.75),
  #     alpha_size_px_max=(2, 8),
  #     alpha_freq_exponent=(-2.5, -2.0),
  #     sparsity=(0.8, 1.0),
  #     density_multiplier=(0.5, 1.0),
  # )
  # rain = iaa.RainLayer(
  #     density=(0.03, 0.14),
  #     density_uniformity=(0.8, 1.0),
  #     drop_size=(0.01, 0.02),
  #     drop_size_uniformity=(0.2, 0.5),
  #     angle=(-15, 15),
  #     speed=(0.05, 0.20),
  #     blur_sigma_fraction=(0.001, 0.001),
  # )
  gn = iaa.GaussianBlur(sigma=(0.0, 2))
  sp=iaa.SaltAndPepper(0.03, per_channel=True)
  for i in range(1500):
      img_pth=img_list[i]
      image_name = os.path.splitext(os.path.basename(img_pth))[0]
      img_s = Image.open(img_pth)
      # img_s.save(os.path.join(output_dir,'src',image_name+'.png'))
      # img_t = brush_stroke_mask(img_s)#TODO
      if i %2==1:
          img_t_aug = gn(image=np.array(img_s))
          img_t = Image.fromarray(np.uint8(img_t_aug))
      else:
          img_t_aug = sp(image=np.array(img_s))
          img_t = Image.fromarray(np.uint8(img_t_aug))
      save_pth=os.path.join(output_dir,'trg-gnsp',image_name+'.png') #todo
      img_t.save(save_pth)
      file_cont.append(save_pth+'\n')
  file_handle.writelines(file_cont)
  file_handle.close()

if __name__ == '__main__':
  main()
#todo:  dataroot outputroot os.mkdir  img_t save_pth