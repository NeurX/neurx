defmodule EvaluateTest do
  use ExUnit.Case
  doctest Neurx.Evaluate

  Code.require_file("test/test_data.exs")
  alias Neurx.{Network, Layer, Neuron, Activators, LossFunctions, Optimizers}
  
  ###############################################
  # Evaluate Endpoint Tests
  ###############################################

  test "Nil PID and Data" do
    try do
      Neurx.evaluate(nil, nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "PID with Nil Data" do
    try do
      # Don't need to actual pass a PID.
      Neurx.evaluate(12345, nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Data with Nil PID" do
    try do
      Neurx.evaluate(nil, %{})
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Testing and Evaluating network training with no hidden layers" do
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
    training_data = TestData.get_simple_training_data()
    options = %{
      epochs: 1000,
      log_freq: 100
    }
    {tnn, final_error} = Neurx.train(nn, training_data, options)

    assert(tnn)
    assert(final_error)
    assert(final_error < 0.01)

    # Evaluating on simple dataset.
    testing_data = TestData.get_simple_test_data()
    {enn, outputs} = Neurx.evaluate(nn, testing_data)
    
    assert(enn)
    assert(outputs)

    # We only provided one testing sample.
    assert(length(outputs) == 1)

    # The testing sample only has one output.
    assert(length(Enum.at(outputs, 0)) == 1)

    # Network should evaluate close to 1.
    assert(Enum.at(Enum.at(outputs, 0), 0) > 0.95)
  end

  test "Testing and Evaluating network training with one hidden layers" do
    # Building the network.
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1
      },
      hidden_layers: [
        %{
          size: 2
        }
      ]
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.output_layer)
    assert(length(network.hidden_layers) == 1)

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
    
    Enum.each(network.hidden_layers, fn lpid ->
      layer = Layer.get(lpid)
      assert(layer)
      assert(length(layer.neurons) == 3)
      assert(layer.activation_fn == sigmoid)
      Enum.each(layer.neurons, fn npid ->
        neuron = Neuron.get(npid)
        assert(neuron)
        assert(neuron.learning_rate == 0.1)
        assert(neuron.optim_fn == sgd)
        if neuron.bias? do
          assert(neuron.activation_fn == nil)
        else
          assert(neuron.activation_fn == sigmoid)
        end
      end)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.learning_rate == 0.1)
    end)

    # Training on simple dataset.
    training_data = TestData.get_simple_training_data()
    options = %{
      epochs: 1000,
      log_freq: 100
    }
    {tnn, final_error} = Neurx.train(nn, training_data, options)

    assert(tnn)
    assert(final_error)
    assert(final_error < 0.01)
    
    # Evaluating on simple dataset.
    testing_data = TestData.get_simple_test_data()
    {enn, outputs} = Neurx.evaluate(nn, testing_data)
    
    assert(enn)
    assert(outputs)

    # We only provided one testing sample.
    assert(length(outputs) == 1)

    # The testing sample only has one output.
    assert(length(Enum.at(outputs, 0)) == 1)

    # Network should evaluate close to 1.
    assert(Enum.at(Enum.at(outputs, 0), 0) > 0.95)
  end
end
