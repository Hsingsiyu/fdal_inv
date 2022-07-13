
import os
import argparse
from tqdm import tqdm
import numpy as np

from utils.inverter import StyleGANInverter
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', default='styleganinv_ffhq256-src',type=str, help='Name of the GAN model.')
  parser.add_argument('--image_list', type=str,default='/home/xsy/datasets/evaluationt_img/celebAHQ1500.txt',
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/{model_nameIMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  parser.add_argument('--E_type', type=str, default='ours',
                      help='baseline or ours')
  parser.add_argument('--netE', type=str, default='/home/xsy/invganV2/fganInv/output/fDAL-stylegan2-ffhq-256Jul11_09-53_stylegan2-ffhq-256_bs_6_epoch3000_regcoef2_0.005_1_adamTrue_DIV_pearson/save_models/styleganinv_encoder_epoch_055.pth',
                      help='path of encoder')
  parser.add_argument('--netG', type=str, default='/home/xsy/invganV2/fganInv/models/pretrain/stylegan2-ffhq-256.pt',
                      help='path of encoder')
  return parser.parse_args()

def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  # assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  net_name=os.path.splitext(os.path.basename(args.netE))[0]

  output_dir = args.output_dir or f'./results/inversion_{args.E_type}/{image_list_name}_{args.model_name}_{net_name}'
  os.makedirs(output_dir)
  os.makedirs(os.path.join(output_dir,'inverted_img'))

  print(f'Loading model.')
  inverter = StyleGANInverter(args.model_name,opt=args)
  image_size = args.viz_size

  # Load image list.
  print(f'Loading image list.')

  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())

  # Initialize visualizer.
  save_interval = 0
  num_iterations=0
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)
  # TODO  inference time?
  # Invert images.
  print(f'Start inversion.')
  latent_codes = []
  for img_idx in tqdm(range(len(image_list)), leave=False):
    image_path = image_list[img_idx]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = resize_image(load_image(image_path), (image_size, image_size))
    code, viz_results=inverter.easy_inti_code(image)
    latent_codes.append(code)
    save_image(f'{output_dir}/{image_name}_ori.png', image)
    save_image(f'{output_dir}/inverted_img/{image_name}_enc.png', viz_results[0])

    visualizer.set_cell(img_idx, 0, text=image_name)
    visualizer.set_cell(img_idx, 1, image=image)
    visualizer.set_cell(img_idx, 2, image=viz_results[0])


  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()