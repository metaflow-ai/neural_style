from .ConvolutionTranspose2D import ConvolutionTranspose2D
from .ATrousConvolution2D import ATrousConvolution2D
from .ScaledSigmoid import ScaledSigmoid
from .PhaseShift import PhaseShift
from .InstanceNormalization import InstanceNormalization

custom_objects={
    'ATrousConvolution2D': ATrousConvolution2D,
    'ConvolutionTranspose2D': ConvolutionTranspose2D,
    'ScaledSigmoid': ScaledSigmoid,
    'PhaseShift': PhaseShift,
    'InstanceNormalization': InstanceNormalization,
}