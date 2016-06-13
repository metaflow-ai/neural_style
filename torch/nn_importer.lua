require 'torch'
require 'nn'
require 'nngraph'
require 'image'

require 'hdf5'
require 'json'

--------------------------------------------------------------------------------
local cmd = torch.CmdLine()

-- Basic options
cmd:option('-output_name', 'my_model.dat')
cmd:option('-model_folder', '../tests/fixture/model_batchnorm')

local function main(params)
  -- Import Keras strucutre
  local model = loadModel(params.model_folder)

  -- Save model
  torch.save('models/' .. params.output_name, model)
end
  
function loadModel(model_folder)
  local archi, weights = loadData(model_folder)  
  
  local node = nil
  local nodes = {}
  local current_layer = nil
  for key, layer in pairs(archi.config.layers) do
    print(layer.class_name .. '.' .. layer.name)
    if key == 1 and not layer.class_name == 'InputLayer' then
      error('First layer of a model should be an input layer, found "%s"' % layer.class_name)
    end

    if layer.class_name == 'InputLayer' then
      batchShapeInput = {
        layer.config.batch_input_shape[1],
        layer.config.batch_input_shape[2],
        layer.config.batch_input_shape[3]
      }
      current_layer = nn.Identity()
      nInputPlane = batchShapeInput[1]
    elseif layer.class_name == 'Convolution2D' then
      current_layer = buildConvolution2D(nInputPlane, layer, weights)
      nInputPlane = layer.config.nb_filter
    elseif layer.class_name == 'ConvolutionTranspose2D' then
      current_layer = buildConvolutionTranspose2D(nInputPlane, layer, weights)
      nInputPlane = layer.config.nb_filter
    elseif layer.class_name == 'BatchNormalization' then
      current_layer = buildBatchNormalization(nInputPlane, layer, weights)
    elseif layer.class_name == 'Merge' then
      if layer.config.mode == 'sum' then
        current_layer = nn.CAddTable()
      end
    elseif layer.class_name == 'Activation' then
      current_layer = buildActivation(layer)
    else
      current_layer = buildCustom(layer)
    end
    

    if #layer.inbound_nodes  == 0 then
      print('No inbound nodes')
      node = current_layer()
    elseif #layer.inbound_nodes == 1 then
      local inboundTables = layer.inbound_nodes[1]
      if #inboundTables == 1 then
        local name = inboundTables[1][1]
        if not nodes[name] then
          error('layer name "%s" not found' % name)
        end
        print('One inbound node: ' .. name)
        node = current_layer(nodes[name])
      else
        print('One input node, multiple table inputs')
        local subNodes = {}
        for key, inboundTable in pairs(inboundTables) do
          name = inboundTable[1]
          if not nodes[name] then
            error('layer name "%s" not found' % name)
          end
          table.insert(subNodes, nodes[name])
        end
        node = current_layer(subNodes)
      end 
    else
      print('Multiple input nodes')
      for inbound_node in layer.inbound_nodes do
        name = inbound_node[1][1]
        subNodes = {}
        if not nodes[name] then
          error('layer name "%s" not found' % name)
        end
        table.insert(subNodes, nodes[name])
      end
      node = current_layer(subNodes)
    end

    if not layer.config.activation == 'linear' then
      node = buildActivation(layer)(node)
    end

    -- Add the node to the dictionnary    
    nodes[layer.name] = node
  end

  -- Get inputs
  local inputs = {}
  for key, layer in pairs(archi.config.input_layers) do
    local name = layer[1]
    if not nodes[name] then
      error('Input layer name "%s" not found' % name)
    end
    table.insert(inputs, nodes[name])
  end

  -- Get outputs
  local outputs = {}
  for key, layer in pairs(archi.config.output_layers) do
    local name = layer[1]
    if not nodes[name] then
      error('Output layer name "%s" not found' % name)
    end
    table.insert(outputs, nodes[name])
  end

  return nn.gModule(inputs, outputs)
end

function loadData(modelFolder)
  local archiFilename = 'archi.json'
  local bestWeightFilename = 'best_weights.hdf5'
  local lastWeightFilename = 'last_weights.hdf5'
  
  local archi, weights
  if file_exists(modelFolder .. '/' .. archiFilename) then
    archi = json.load(modelFolder .. '/' .. archiFilename)
    if archi.config.layers[1].config.input_dtype == 'float32' then
      torch.setdefaulttensortype('torch.FloatTensor')
    elseif archi.config.layers[1].config.input_dtype == 'float64' then
      torch.setdefaulttensortype('torch.DoubleTensor')
    else
      error('Tensor type "%s" not supported yet' % archi.config.layers[1].config.input_dtype)
    end
  else
    error('Model architecture file "%s" not found in folder "%s"' % { archiFilename, modelFolder })
  end

  if file_exists(modelFolder .. '/' .. bestWeightFilename) then
    local weightsFile = hdf5.open(modelFolder .. '/' .. bestWeightFilename, 'r')
    weights = weightsFile:read():all()
  elseif file_exists(modelFolder .. '/' .. lastWeightFilename) then
    local weightsFile = hdf5.open(modelFolder .. '/' .. lastWeightFilename, 'r')
    weights = weightsFile:read():all()
  else
    error('Model architecture file "%s"/"%s" not found in folder "%s"' % { archiFilename, modelFolder })
  end  

  return archi, weights
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function buildConvolution2D(nInputPlane, layer, weights)
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
  local net_layer = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW)

  -- Loading weights
  local weight = weights[layer.name][layer.name .. "_W"]:float()
  local bias = weights[layer.name][layer.name .. "_b"]:float()
  -- We need to reverse the matrix weight to perform the exact same calculation
  -- in torch and in theano
  reversedWeight = image.flip(weight, 3)
  reversedWeight = image.flip(reversedWeight, 4)
  net_layer.weight = reversedWeight
  net_layer.bias = bias

  return net_layer
end

function buildConvolutionTranspose2D(nInputPlane, layer, weights)
  nOutputPlane = layer.config.nb_filter
  kW = layer.config.nb_col
  kH = layer.config.nb_row
  dW = layer.config.subsample[1]
  dH = layer.config.subsample[2]
  if layer.config.border_mode == "same" then
    padW = (kW - 1) / 2
    adjW = 1
  else
    padW = 0
    adjW = 0
  end
  padH = padW
  adjH = adjW
  local net_layer = nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

  -- Loading weights
  local weight = weights[layer.name][layer.name .. "_W"]:float()
  local bias = weights[layer.name][layer.name .. "_b"]:float()
  -- We need to reverse the matrix weight to perform the exact same calculation
  -- in torch and in theano
  reversedWeight = image.flip(weight, 3)
  reversedWeight = image.flip(reversedWeight, 4)
  net_layer.weight = reversedWeight
  net_layer.bias = bias

  return net_layer
end

function buildBatchNormalization(nInputPlane, layer, weights)
  local net_layer = nn.SpatialBatchNormalization(nInputPlane, layer.config.epsilon, layer.config.momentum, layer.config.trainable)

  -- Loading weights
  net_layer.bias = weights[layer.name][layer.name .. "_beta"]:float()
  net_layer.weight = weights[layer.name][layer.name .. "_gamma"]:float()
  local std = weights[layer.name][layer.name .. "_running_std"]:float()
  net_layer.running_var = torch.Tensor(std):fill(1):cdiv(torch.cmul(std, std))
  net_layer.running_mean = weights[layer.name][layer.name .. "_running_mean"]:float()

  return net_layer
end

function buildActivation(layer)
  if layer.config.activation == 'relu' then
    return nn.ReLU()
  elseif layer.config.activation == '<lambda>' then
    
  end
end

function buildCustom(layer)
  if layer.class_name == 'ScaledSigmoid' then
    return function (node)
      node = nn.MulConstant(1 / layer.config.scaling)(node)
      node = nn.Sigmoid()(node)
      node = nn.MulConstant(layer.config.scaling)(node)
      return node
    end
  else
    error('Can\'t load classname "%s"' % layer.class_name)
  end
end

local params = cmd:parse(arg)
main(params)