import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
'''
This file allows for images to be augmented according to the
CIFAR implementation of SimCLR. To augment a batch of images, 
pass the batch to augment(). To efficiently augment a batch
twice (needed for SimCLR training), call augment with the
optional parameter augment_twice=True. This will return
a list of tensors, which can be unpacked as:
aug1, aug2 = augment(batch, True)

To visualize how augmentation works, call visualize(). 
'''

def augment(images, augment_twice=False):
  num_images = len(images)

  # define augmentation
  augmentation = transforms.Compose([
      _get_random_crop(),
      _get_random_flip(),
      *_get_color_distortion(s=0.5)
  ])

  if augment_twice:
    images = images.repeat((2, 1, 1, 1))

  # perform augmentation
  transformation_fnc = transforms.Lambda(lambda batch: torch.stack([augmentation(image) for image in batch]))
  transformed_images = transformation_fnc(images)
  if augment_twice:
    return transformed_images.split(num_images)
  return transformation_fnc(images)

def visualize(size=5, num_transformations=2, show=True):
  batch = _get_batch(size=size)
  transformed_batches = [augment(batch) for _ in range(num_transformations)]
  if show:
    for images in zip(batch, *transformed_batches):
      _display_images(images)

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

  for inputs, _ in train_loader:
    return inputs

def _display_images(images):
  fig, axis = plt.subplots(1, len(images))
  for idx, (ax, image) in enumerate(zip(axis, images)):
    ax.imshow(image.permute(1,2,0))
    if idx == 0:
      ax.set_title('original')
    else:
      ax.set_title("aug {:d}".format(idx-1))
  plt.show()
