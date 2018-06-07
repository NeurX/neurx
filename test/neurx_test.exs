defmodule NeurxTest do
  use ExUnit.Case
  doctest Neurx

  alias Neurx.{Network, Layer, Neuron, Activators, LossFunctions, Optimizers}

  # TODO: Make it so the config actually sets the learning rate.
  # TODO: Test the learning rate is being set.

  test "Create basic 3:1 network." do
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1,
        activation: %{
          type: "Sigmoid"
        }
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.output_layer)
    assert(network.hidden_layers == [])
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)

    sigmoid = Activators.retreiveFunction("Sigmoid")
    mse = LossFunctions.retreiveFunction("MSE")
    sgd = Optimizers.retreiveFunction("SGD")
    assert(sigmoid)
    assert(mse)
    assert(sgd)
    
    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
    end)
  end
  
  test "Create basic 3:2:1 network." do
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1,
        activation: %{
          type: "Sigmoid"
        }
      },
      hidden_layers: [
        %{
          size: 2
        }
      ],
      loss_function: %{
        type: "MSE"
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.hidden_layers)
    assert(length(network.hidden_layers) == 1)
    assert(network.output_layer)
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)
    
    sigmoid = Activators.retreiveFunction("Sigmoid")
    mse = LossFunctions.retreiveFunction("MSE")
    sgd = Optimizers.retreiveFunction("SGD")
    assert(sigmoid)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)

    Enum.each(network.hidden_layers, fn lpid ->
      layer = Layer.get(lpid)
      assert(layer)
      assert(length(layer.neurons) == 3)
      assert(layer.activation_fn == sigmoid)
      Enum.each(layer.neurons, fn npid ->
        neuron = Neuron.get(npid)
        assert(neuron)
        if neuron.bias? do
          assert(neuron.activation_fn == nil)
        else
          assert(neuron.activation_fn == sigmoid)
        end
      end)
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
    end)
  end
  
  test "Create 100:50:25:2 network." do
    nn = Neurx.build(%{
      input_layer: 100,
      output_layer: %{
        size: 2
      },
      hidden_layers: [
        %{
          size: 50,
          activation: %{
            type: "Relu"
          }
        },
        %{
          size: 25,
          activation: %{
            type: "Relu"
          }
        }
      ],
      loss_function: %{
        type: "MSE"
      },
      optimization_function: %{
        type: "SGD",
        learning_rate: 0.3
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.hidden_layers)
    assert(length(network.hidden_layers) == 2)
    assert(network.output_layer)
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)
    
    relu = Activators.retreiveFunction("Relu")
    sigmoid = Activators.retreiveFunction("Sigmoid")
    mse = LossFunctions.retreiveFunction("MSE")
    sgd = Optimizers.retreiveFunction("SGD")
    assert(relu)
    assert(sigmoid)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 101)
    assert(input_layer.activation_fn == nil)

    hl1 = Enum.at(network.hidden_layers, 0)
    hlayer1 = Layer.get(hl1)
    assert(hlayer1)
    assert(length(hlayer1.neurons) == 51)
    assert(hlayer1.activation_fn == relu)
    Enum.each(hlayer1.neurons, fn npid ->
      neuron = Neuron.get(npid)
      assert(neuron)
      if neuron.bias? do
        assert(neuron.activation_fn == nil)
      else
        assert(neuron.activation_fn == relu)
      end
    end)

    hl2 = Enum.at(network.hidden_layers, 1)
    hlayer2 = Layer.get(hl2)
    assert(hlayer2)
    assert(length(hlayer2.neurons) == 26)
    assert(hlayer2.activation_fn == relu)
    Enum.each(hlayer2.neurons, fn npid ->
      neuron = Neuron.get(npid)
      assert(neuron)
      if neuron.bias? do
        assert(neuron.activation_fn == nil)
      else
        assert(neuron.activation_fn == relu)
      end
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 2)
    assert(output_layer.activation_fn == sigmoid)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
    end)
  end
end
