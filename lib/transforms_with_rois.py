from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,rois):
        for t in self.transforms:
            if 'RandomSizedCrop' in t.__str__() \
                    or 'CenterCrop' in t.__str__() :

                img,rois = t(img,rois)
                rois = FixRois(img.size, rois)
            elif 'RandomHorizontalFlip' in t.__str__() \
                    or 'Scale' in t.__str__():

                img,rois = t(img,rois)
            else:
                img = t(img)

        # for vgg16
        # rois = OffSet(rois, (img.shape[2],img.shape[1]), o0=8.5, o=9.5, stride=[16,16])

        return img, rois


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class ToPILImage(object):
    """Convert a tensor to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving the value range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.

        Returns:
            PIL.Image: Image converted to PIL.Image.

        """
        npimg = pic
        mode = None
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
            if npimg.dtype == np.int16:
                mode = 'I;16'
            if npimg.dtype == np.int32:
                mode = 'I'
            elif npimg.dtype == np.float32:
                mode = 'F'
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode=mode)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR, scaleheight=None):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.scaleheight = scaleheight

    def __call__(self, img, rois):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if self.scaleheight is not None:
            for attempt in range(10):
                oh = self.scaleheight[random.randint(0,len(self.scaleheight)-1)]
                ow = int(img.size[0]/img.size[1]*oh)

                if oh<=700 & ow <=700:
                    return img.resize((ow, oh), self.interpolation), ResizeRois(img.size, (ow, oh), rois)

            ow = 650#700
            oh = int(img.size[1]/img.size[0]*ow)
            return img.resize((ow, oh), self.interpolation), ResizeRois(img.size, (ow, oh), rois)
            # for attempt in range(10):
            #     if img.size[0]<img.size[1]:
            #         oh = self.scaleheight[random.randint(0,len(self.scaleheight)-1)]
            #         ow = int(img.size[0]/img.size[1]*oh)
            #     else:
            #         ow = self.scaleheight[random.randint(0,len(self.scaleheight)-1)]
            #         oh = int(img.size[1]/img.size[0]*ow)
            #
            #     return img.resize((ow, oh), self.interpolation), ResizeRois(img.size, (ow, oh), rois)
            #
            # return img, rois
        else:
            if isinstance(self.size, int):
                w, h = img.size
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    return img, rois
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    return img.resize((ow, oh), self.interpolation), ResizeRois(img.size, (ow, oh), rois)
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    return img.resize((ow, oh), self.interpolation), ResizeRois(img.size, (ow, oh), rois)
            else:
                return img.resize(self.size, self.interpolation), ResizeRois(img.size, self.size, rois)


class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, rois):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), RemoveOuterRois((x1, y1, x1 + tw, y1 + th), rois)


class Pad(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        raise NotImplementedError

        return ImageOps.expand(img, border=self.padding, fill=self.fill)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, rois):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            raise NotImplementedError


        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img,rois

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), RemoveOuterRois((x1, y1, x1 + tw, y1 + th), rois)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, rois):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            rois[:,[1,3]] = img.size[0] + 1 - rois[:,[3,1]];
            return img.transpose(Image.FLIP_LEFT_RIGHT), rois
        return img, rois

class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img,rois):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                rois = RemoveOuterRois((x1, y1, x1 + w, y1 + h), rois)
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation), ResizeRois(img.size, (self.size,self.size),rois)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)

        img,rois = scale(img,rois)
        return crop(img,rois)

def RemoveOuterRois(crop, rois):# remove rois out of bounding and move to new coordinate, e.g. use after crop immidiately
    x1, y1, x2, y2 = crop

    rois[:,1] = torch.max(torch.FloatTensor([1]), rois[:,1]-x1+1)# might be inaccuract due to crop interpolation
    rois[:,2] = torch.max(torch.FloatTensor([1]), rois[:,2]-y1+1)
    rois[:,3] = torch.min(torch.FloatTensor([x2-x1]), rois[:,3]-x1+1)
    rois[:,4] = torch.min(torch.FloatTensor([y2-y1]), rois[:,4]-x1+1)

    return rois

def ResizeRois(sizeIn, sizeOut, rois):# resize rois according to image transforms, e.g. use after resize immidiately
    if isinstance(sizeIn, numbers.Number):
            inw,inh = (int(sizeIn), int(sizeIn))
    else:
        inw,inh = sizeIn
    if isinstance(sizeOut, numbers.Number):
            outw,outh = (int(sizeOut), int(sizeOut))
    else:
        outw,outh = sizeOut

    # relative box center and width/hegiht, [index x1(horizonal) y1(vertical) x2 y2]
    bxr = (rois[:,1] + rois[:,3])/2/inw
    byr = (rois[:,2] + rois[:,4])/2/inh
    bwr = (rois[:,3] - rois[:,1])/inw
    bhr = (rois[:,4] - rois[:,2])/inh

    # new relative box center and width/hegiht
    bxnew = outw*bxr
    bynew = outh*byr
    bwnew = outw*bwr
    bhnew = outh*bhr

    rois[:,1] = torch.max(torch.FloatTensor([1]), torch.round(bxnew - bwnew/2))
    rois[:,2] = torch.max(torch.FloatTensor([1]), torch.round(bynew - bhnew/2))
    rois[:,3] = torch.min(torch.FloatTensor([outw]), torch.round(bxnew + bwnew/2))
    rois[:,4] = torch.min(torch.FloatTensor([outh]), torch.round(bynew + bynew/2))

    return rois

def FixRois(size, rois):# remove meaningless rois,due to 'crop', minrois is defined according to current cordinate

    # rois_ = np.concatenate((rois.numpy(),np.array([[0,1,1,size[0],size[1]]])),axis=0)
    rois_ = rois.numpy()

    isvalid = np.where((rois_[:,1]>=1) & (rois_[:,2]>=1) & \
    (rois_[:,1]<rois_[:,3]) & (rois_[:,2]<rois_[:,4]) & \
    (rois_[:,3]<=size[0]) & (rois_[:,4]<=size[1]))

    return torch.from_numpy(np.unique(rois_[isvalid],axis=0)).type(torch.FloatTensor)

def OffSet(rois, size, o0=8.5, o=9.5, stride=[16,16]):
    # x1(horizonal,width) y1(vertical,height) x2 y2

    rois[:,1] += (-o0+o + stride[0]*0.5)
    rois[:,2] += (-o0+o + stride[0]*0.5)
    rois[:,3] += (-o0-o - stride[1]*0.5)
    rois[:,4] += (-o0-o - stride[1]*0.5)

    return FixRois(size, rois)

