import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
'''
This file allows for images to be augmented according to the
CIFAR implementation of SimCLR. To augment a batch of images, 
pass the batch to augment(). Augmenting a batch will augment 
each image in the batch twice, according to Algorithm 1 from 
the SimCLR paper. 

To visualize how this is working, call visualize(). 
'''

def augment(images):
  # define augmentation
  augmentation = transforms.Compose([
      _get_random_crop(),
      _get_random_flip(),
      *_get_color_distortion(s=0.5)
  ])

  # perform augmentation
  transformation_fnc = transforms.Lambda(lambda images: torch.stack([augmentation(image) for image in images])) 
  first_transformation = transformation_fnc(images)
  second_transformation = transformation_fnc(images)

  # assemble augmentations
  original_size = first_transformation.size()
  size = (2*original_size[0], *original_size[1:])
  x = torch.empty(size)
  for idx, (first, second) in enumerate(zip(first_transformation, second_transformation)):
    x[2*idx] = first
    x[2*idx+1] = second
  return x

def visualize(size=5, show=True):
  images = _get_batch(size=size)
  transformed_images = augment(images)
  if show:
    if len(images) == len(transformed_images):
      for base, transformed in zip(images, transformed_images):
        _display_images([base, transformed])
    elif 2*len(images) == len(transformed_images):
      for base, trans1, trans2 in zip(images, transformed_images[::2], transformed_images[1::2]):
        _display_images([base, trans1, trans2])


def _get_random_crop():
  return transforms.RandomResizedCrop(32)

def _get_random_flip():
  return transforms.RandomHorizontalFlip()

# method get_color_distortion is from SimCLR paper (Appendix A)
def _get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return [rnd_color_jitter, rnd_gray]

def _get_batch(size=128):
  transform_train = transforms.Compose([
      transforms.ToTensor(),
  ])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=size, shuffle=True, num_workers=16)

  for batch_idx, (inputs, targets) in enumerate(train_loader):
    return inputs

def _display_images(images):
  fig, ax = plt.subplots(1, len(images))
  for idx in range(len(images)):
    ax[idx].imshow(images[idx].permute(1,2,0))
  plt.show()
