
from tqdm import tqdm
import cv2
import numpy as np

import torch
#todo
from models.stylegan import Generator
from models.e2style_encoder import BackboneEncoderFirstStage


__all__ = ['StyleGANInverter']
DTYPE_NAME_TO_TORCH_TENSOR_TYPE = {
    'float16': torch.HalfTensor,
    'float32': torch.FloatTensor,
    'float64': torch.DoubleTensor,
    'int8': torch.CharTensor,
    'int16': torch.ShortTensor,
    'int32': torch.IntTensor,
    'int64': torch.LongTensor,
    'uint8': torch.ByteTensor,
    'bool': torch.BoolTensor,
}

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()

class StyleGANInverter(object):
  def __init__(self,model_name,opt,):
    self.model_name = model_name
    ckptG = torch.load(opt.netG, map_location=torch.device('cpu'))
    self.G = Generator(size=opt.viz_size,style_dim=512,n_mlp=8).cuda()
    self.G.load_state_dict(ckptG["g_ema"], strict=False)
    self.G.eval()
    if opt.E_type=='ours':
      self.E = BackboneEncoderFirstStage(image_size=opt.viz_size, num_layers=50, mode='ir_se').cuda()
    ckptE = torch.load(opt.netE, map_location=torch.device('cpu'))
    self.E.load_state_dict({k.replace('module.', ''): v for k, v in ckptE.items()})
    self.E.eval()
    # assert list(self.encode_dim) == list(self.E.encode_dim)
    self.run_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.max_val=1
    self.min_val=-1
    self.image_channels=3
  def preprocess(self, image):
    """Preprocesses a single image.

    This function assumes the input numpy array is with shape [height, width,channel],
    channel order `RGB`, and pixel range [0, 255].

    The returned image is with shape [channel, new_height, new_width], where
    `new_height` and `new_width` are specified by the given generative model.
    The channel order of returned image is also specified by the generative
    model. The pixel range is shifted to [min_val, max_val], where `min_val` and
    `max_val` are also specified by the generative model.
    """
    if not isinstance(image, np.ndarray):
      raise ValueError(f'Input image should be with type `numpy.ndarray`!')
    if image.dtype != np.uint8:
      raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

    if image.ndim != 3 or image.shape[2] not in [1, 3]:
      raise ValueError(f'Input should be with shape [height, width, channel], '
                       f'where channel equals to 1 or 3!\n'
                       f'But {image.shape} is received!')
    # if image.shape[2] == 1 and self.G.image_channels == 3:
    #   image = np.tile(image, (1, 1, 3))
    # if image.shape[2] != self.G.image_channels:
    #   raise ValueError(f'Number of channels of input image, which is '
    #                    f'{image.shape[2]}, is not supported by the current '
    #                    f'inverter, which requires {self.G.image_channels} '
    #                    f'channels!')

    image = image.astype(np.float32)
    image = image / 255.0 * (self.max_val - self.min_val) + self.min_val
    image = image.astype(np.float32).transpose(2, 0, 1) #[3,w,h]

    return image

  def postprocess(self, images):
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

    if images.ndim != 4 or images.shape[1] != self.image_channels:
      raise ValueError(f'Input should be with shape [batch_size, channel, '
                       f'height, width], where channel equals to '
                       f'{self.image_channels}!\n'
                       f'But {images.shape} is received!')
    images = (images - self.min_val) * 255 / (self.max_val - self.min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    # if self.image_channels == 3 and self.channel_order == 'BGR':
      # images = images[:, :, :, ::-1]

    return images
  def to_tensor(self, array):
    dtype = type(array)
    if isinstance(array, torch.Tensor):
      tensor = array
    elif isinstance(array, np.ndarray):
      tensor_type = DTYPE_NAME_TO_TORCH_TENSOR_TYPE[array.dtype.name]
      tensor = torch.from_numpy(array).type(tensor_type)
    else:
      raise ValueError(f'Unsupported input type `{dtype}`!')
    tensor = tensor.to(self.run_device)
    return tensor
  def get_init_code(self, image):
    """Gets initial latent codes as the start point for optimization.

    The input image is assumed to have already been preprocessed, meaning to
    have shape [self.G.image_channels, self.G.resolution, self.G.resolution],
    channel order `self.G.channel_order`, and pixel range [self.G.min_val,
    self.G.max_val].
    """
    x = image[np.newaxis]
    x = self.to_tensor(x.astype(np.float32))
    # print(x.shape)
    # if self.E_type == 'baseline':
    #   z = self.E.net(x)
    # else:
    z = self.E(x)
    z = _get_tensor_value(z.view(1, z.shape[1],z.shape[2]))
    return z.astype(np.float32)

  def easy_inti_code(self,image):
    viz_results = []
    init_z = self.get_init_code(self.preprocess(image))
    z = torch.Tensor(init_z).to(self.run_device)
    x_init_inv, _  = self.G([z], input_is_latent=True, randomize_noise=False, return_latents=False)
    viz_results.append(self.postprocess(_get_tensor_value(x_init_inv))[0])
    z = _get_tensor_value(z)
    return z.astype(np.float32),viz_results

  def easy_invert(self, image, num_viz=0):
    """Wraps functions `preprocess()` and `invert()` together."""
    return self.easy_inti_code(self.preprocess(image), num_viz)

  # def diffuse(self,
  #             target,
  #             context,
  #             center_x,
  #             center_y,
  #             crop_x,
  #             crop_y,
  #             num_viz=0):
  #   """Diffuses the target image to a context image.
  #
  #   Basically, this function is a motified version of `self.invert()`. More
  #   concretely, the encoder regularizer is removed from the objectives and the
  #   reconstruction loss is computed from the masked region.
  #
  #   Args:
  #     target: Target image (foreground).
  #     context: Context image (background).
  #     center_x: The x-coordinate of the crop center.
  #     center_y: The y-coordinate of the crop center.
  #     crop_x: The crop size along the x-axis.
  #     crop_y: The crop size along the y-axis.
  #     num_viz: Number of intermediate outputs to visualize. (default: 0)
  #
  #   Returns:
  #     A two-element tuple. First one is the inverted code. Second one is a list
  #       of intermediate results, where first image is the direct copy-paste
  #       image, second one is the reconstructed result from the initial latent
  #       code, remainings are from the optimization process every
  #       `self.iteration // num_viz` steps.
  #   """
  #   image_shape = (self.G.image_channels, self.G.resolution, self.G.resolution)
  #   mask = np.zeros((1, *image_shape), dtype=np.float32)
  #   xx = center_x - crop_x // 2
  #   yy = center_y - crop_y // 2
  #   mask[:, :, yy:yy + crop_y, xx:xx + crop_x] = 1.0
  #
  #   target = target[np.newaxis]
  #   context = context[np.newaxis]
  #   x = target * mask + context * (1 - mask)
  #   x = self.G.to_tensor(x.astype(np.float32))
  #   x.requires_grad = False
  #   mask = self.G.to_tensor(mask.astype(np.float32))
  #   mask.requires_grad = False
  #
  #   init_z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
  #   init_z = init_z.astype(np.float32)
  #   z = torch.Tensor(init_z).to(self.run_device)
  #   z.requires_grad = True
  #
  #   optimizer = torch.optim.Adam([z], lr=self.learning_rate)
  #
  #   viz_results = []
  #   viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
  #   x_init_inv = self.G.net.synthesis(z)
  #   viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
  #   pbar = tqdm(range(1, self.iteration + 1), leave=True)
  #   for step in pbar:
  #     loss = 0.0
  #
  #     # Reconstruction loss.
  #     x_rec = self.G.net.synthesis(z)
  #     loss_pix = torch.mean(((x - x_rec) * mask) ** 2)
  #     loss = loss + loss_pix * self.loss_pix_weight
  #     log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'
  #
  #     # Perceptual loss.
  #     if self.loss_feat_weight:
  #       x_feat = self.F.net(x * mask)
  #       x_rec_feat = self.F.net(x_rec * mask)
  #       loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
  #       loss = loss + loss_feat * self.loss_feat_weight
  #       log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'
  #
  #     log_message += f', loss: {_get_tensor_value(loss):.3f}'
  #     pbar.set_description_str(log_message)
  #     if self.logger:
  #       self.logger.debug(f'Step: {step:05d}, '
  #                         f'lr: {self.learning_rate:.2e}, '
  #                         f'{log_message}')
  #
  #     # Do optimization.
  #     optimizer.zero_grad()
  #     loss.backward()
  #     optimizer.step()
  #
  #     if num_viz > 0 and step % (self.iteration // num_viz) == 0:
  #       viz_results.append(self.G.postprocess(_get_tensor_value(x_rec))[0])
  #
  #   return _get_tensor_value(z), viz_results
  #
  # def easy_diffuse(self, target, context, *args, **kwargs):
  #   """Wraps functions `preprocess()` and `diffuse()` together."""
  #   return self.diffuse(self.preprocess(target),
  #                       self.preprocess(context),
  #                       *args, **kwargs)
