defmodule Neurx.LossFunctionsTest do
  use ExUnit.Case
  doctest Neurx.LossFunctions

  alias Neurx.{LossFunctions, Network, Layer, Neuron}
  
  test "Testing function retrieval." do
    assert(LossFunctions.getFunction("MSE"))
    try do
      assert(LossFunctions.getFunction("REKT"))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
    try do
      assert(LossFunctions.getFunction(nil))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Testing Mean Squared Error." do
    nn = Neurx.build(%{
      input_layer: 4,
      output_layer: %{
        size: 2
      }
    })

    assert(nn)
    network = Network.get(nn)

    assert(network.output_layer)
    output_neurons = Layer.get(network.output_layer).neurons
    assert(output_neurons)

    mse = LossFunctions.mean_squared_error(output_neurons, [1, 2])
    assert(mse == 2.5)
  end
end
