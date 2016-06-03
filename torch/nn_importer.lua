require 'torch'
require 'nn'
require 'image'

require 'hdf5'
require 'json'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-content_image', '../data/test/COCO_test2014_000000000001.jpg',
           'Content target image')
cmd:option('-gpu', -1, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-output_image', 'out.png')

cmd:option('-archi', '../tests/fixture/model/archi.json')
cmd:option('-weights', '../tests/fixture/model/last_weights.hdf5')
cmd:option('-backend', 'nn', 'nn|clnn')

local function main(params)

  -- Backend conf
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end
  -- END Backend conf

  -- Load ST
  local archi = json.load(params.archi)
  local weightsFile = hdf5.open(params.weights, 'r')
  -- local data = weightsFile:read('/path/to/data'):all()

  -- cnn = 

  -- if params.gpu >= 0 then
  --   if params.backend ~= 'clnn' then
  --     cnn:cuda()
  --   else
  --     cnn:cl()
  --   end
  -- end
  -- -- END Load ST
  
  
  -- -- Loading content_image
  -- local content_image = image.load(params.content_image, 3)
  -- content_image = image.scgetale(content_image, 600, 'bilinear')
  -- local content_image_processed = preprocess(content_image):float()
  -- -- END Loading content_image
  
  -- -- Initialize the image
  -- if params.gpu >= 0 then
  --   if params.backend ~= 'clnn' then
  --     content_image_processed = content_image_processed:cuda()
  --   else
  --     content_image_processed = content_image_processed:cl()
  --   end
  -- end
  
  -- -- Run it through the network once to get the proper size for the gradient
  -- -- All the gradients will come from the extra loss modules, so we just pass
  -- -- zeros into the top of the net on the backward pass.
  -- local img = net:forward(content_image_processed)


  -- local function save()
  --   local disp = deprocess(img:double())
  --   disp = image.minmax{tensor=disp, min=0, max=1}
  --   local filename = build_filename(params.output_image, t)
  --   image.save(filename, disp)
  -- end

end
  

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
function preprocess(img)
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


local params = cmd:parse(arg)
main(params)