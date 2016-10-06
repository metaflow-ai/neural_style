from .ConvolutionTranspose2D import ConvolutionTranspose2D
from .ATrousConvolution2D import ATrousConvolution2D
from .ScaledSigmoid import ScaledSigmoid
from .PhaseShift import PhaseShift
from .InstanceNormalization import InstanceNormalization
from .ReflectPadding2D import ReflectPadding2D

custom_objects={
    'ATrousConvolution2D': ATrousConvolution2D,
    'ConvolutionTranspose2D': ConvolutionTranspose2D,
    'ScaledSigmoid': ScaledSigmoid,
    'PhaseShift': PhaseShift,
    'InstanceNormalization': InstanceNormalization,
    'ReflectPadding2D': ReflectPadding2D,
}