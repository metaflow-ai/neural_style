require 'torch'
require 'nn'
require 'nngraph'
require 'image'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-input_img', '../tests/fixture/blue.png',
           'Image to be processsed')
cmd:option('-gpu', -1, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-output_image', 'out.png')
cmd:option('-image_size', 3, 'Maximum height / width of generated image')

cmd:option('-model_name', 'models/my_model.dat')

local function main(params)
  -- Load net
  net = torch.load(params.model_name)  

  -- Load input img
  inputImg = loadImg(params.input_img, params.image_size)

  -- Forward input img
  local outImg = net:forward(inputImg)
  print(inputImg, outImg:int())
  
  -- Sage output img
  saveImg(outImg, params.output_image)
end

function loadImg(path, size)
  local inputImg = image.load(path, 3)
  inputImg = image.scale(inputImg, size, 'bilinear')
  inputImg = preprocess(inputImg)

  return inputImg
end

-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
function preprocess(img)
  local perm = torch.LongTensor{3, 2, 1}
  local dims = #img
  img = img:index(1, perm):mul(255.0):resize(1, dims[1], dims[2], dims[3]):double()
  return img
end

function saveImg(img, filename)
  local disp = deprocess(img:double())
  disp = image.minmax{tensor=disp, min=0, max=1}
  image.save(filename, disp)
end


-- Undo the above preprocessing.
function deprocess(img)
  img = img[1]
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(255.0)
  return img
end


local params = cmd:parse(arg)
main(params)