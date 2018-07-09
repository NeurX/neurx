defmodule TrainTest do
  use ExUnit.Case
  doctest Neurx.Train

  Code.require_file("test/test_data.exs")
  alias Neurx.{Network, Layer, Neuron, Activators, LossFunctions, Optimizers}
  
  ###############################################
  # Build Endpoint Tests
  ###############################################

  test "Nil PID and Data" do
    try do
      Neurx.train(nil, nil, nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "PID with Nil Data" do
    try do
      # Don't need to actual pass a PID.
      Neurx.train(12345, nil, nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Data with Nil PID" do
    try do
      # Don't need to actual pass a PID.
      Neurx.train(nil, %{}, nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Testing Network Training" do
    # Building the network.
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.output_layer)
    assert(network.hidden_layers == [])

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)

    sigmoid = Activators.getFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(sigmoid)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 3 + 1)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)

    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.learning_rate == 0.1)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.learning_rate == 0.1)
    end)

    # Training on simple dataset.
    data = TestData.get_simple_training_data()
    options = %{
      epochs: 1000,
      log_freq: 100
    }
    tnn = Neurx.train(nn, data, options)

    assert(tnn)

    network = Network.get(tnn)
    assert(network)

    assert(network.error < 0.01)
  end
end
