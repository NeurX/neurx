defmodule NeurxTest do
  use ExUnit.Case
  doctest Neurx

  alias Neurx.{Network, Layer, Neuron}

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

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn)
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

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)

    Enum.each(network.hidden_layers, fn lpid ->
      layer = Layer.get(lpid)
      assert(layer)
      assert(length(layer.neurons) == 3)
      assert(layer.activation_fn)
      Enum.each(layer.neurons, fn npid ->
        neuron = Neuron.get(npid)
        assert(neuron)
        assert(neuron.activation_fn)
      end)
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn)
    end)
  end
end
