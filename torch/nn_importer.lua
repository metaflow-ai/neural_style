require 'torch'
require 'nn'

require 'hdf5'
require 'json'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-output_name', 'my_model.dat')
cmd:option('-archi', '../tests/fixture/model_import/archi.json')
cmd:option('-weights', '../tests/fixture/model_import/last_weights.hdf5')

local function main(params)
  -- Import Keras strucutre
  local net = loadModel(params.archi, params.weights)

  -- Save model
  print('Saving model:')
  print(net)
  torch.save('models/' .. params.output_name, net)
end
  
function loadModel(jsonfile, weightsFile)
  local archi = json.load(jsonfile)
  local weightsFile = hdf5.open(weightsFile, 'r')
  local weights = weightsFile:read():all()

  local net = nn.Sequential()
  for key, layer in pairs(archi.config.layers) do
    if key == 1 and not layer.class_name == 'InputLayer' then
      error(string.format('First layer of a model should be an input layer, found "%s"', layer.class_name))
    elseif key == 1 then
      batchShapeInput = {
        layer.config.batch_input_shape[1],
        layer.config.batch_input_shape[2],
        layer.config.batch_input_shape[3]
      }
      nInputPlane = batchShapeInput[1]
    end

    if layer.class_name == 'Convolution2D' then
      local net_layer = buildConvolution2D(nInputPlane, layer)
      local weight = weights[layer.name][layer.name .. "_W"]:double()
      local bias = weights[layer.name][layer.name .. "_b"]:double()
      net_layer.weight = weight
      net_layer.bias = bias
      net:add(net_layer)

      nInputPlane = layer.config.nb_filter
    elseif layer.class_name == 'BatchNormalization' then
      net:add(buildBatchNormalization(nInputPlane, layer))

    elseif layer.class_name == 'ConvolutionTranspose2D' then
      local net_layer = buildConvolutionTranspose2D(nInputPlane, layer)
      local weight = weights[layer.name][layer.name .. "_W"]:double()
      local bias = weights[layer.name][layer.name .. "_b"]:double()
      net_layer.weight = weight
      net_layer.bias = bias
      net:add(net_layer)

      nInputPlane = layer.config.nb_filter
    end
    if (not layer.config.activation == 'linear') or layer.class_name == 'Activation' then
        net:add(buildActivation(layer))
    end
  end

  return net
end

function buildConvolution2D(nInputPlane, layer)
  nOutputPlane = layer.config.nb_filter
  kW = layer.config.nb_col
  kH = layer.config.nb_row
  dW = layer.config.subsample[1]
  dH = layer.config.subsample[2]
  if layer.config.border_mode == "same" then
    padW = (kW - 1) / 2
  else
    padW = 0
  end
  return nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW)
end

function buildConvolutionTranspose2D(nInputPlane, layer)
  nOutputPlane = layer.config.nb_filter
  kW = layer.config.nb_col
  kH = layer.config.nb_row
  dW = layer.config.subsample[1]
  dH = layer.config.subsample[2]
  if layer.config.border_mode == "same" then
    padW = (kW - 1) / 2
  else
    padW = 0
  end
  padH = padW
  return nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end


function buildActivation(layer)
  if layer.config.activation == 'relu' then
    return nn.ReLU()
  end
end

local params = cmd:parse(arg)
main(params)