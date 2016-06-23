from . import ConvolutionTranspose2D
from . import ATrousConvolution2D
from . import ScaledSigmoid

custom_objects={
    'ATrousConvolution2D': ATrousConvolution2D,
    'ConvolutionTranspose2D': ConvolutionTranspose2D,
    'ScaledSigmoid': ScaledSigmoid
}