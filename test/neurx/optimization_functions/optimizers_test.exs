defmodule Neurx.OptimizersTest do
  use ExUnit.Case
  doctest Neurx.Optimizers

  alias Neurx.{Optimizers, Network, Layer, Neuron, Connection}

  test "Testing function retrieval." do
    assert(Optimizers.getFunction("SGD"))
    try do
      assert(Optimizers.getFunction("REKT"))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
    try do
      assert(Optimizers.getFunction(nil))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Test SGD optimization function." do
    nn = Neurx.build(%{
      input_layer: 1,
      output_layer: %{
        size: 1
      }
    })

    assert(nn)
    network = Network.get(nn)

    assert(network.input_layer)
    input_neurons = Layer.get(network.input_layer).neurons
    assert(input_neurons)
    assert(length(input_neurons) == 2)

    input_neuron_pid = Enum.at(input_neurons, 1)
    assert(input_neuron_pid)
    input_neuron = Neuron.get(input_neuron_pid)
    assert(input_neuron)
    assert(length(input_neuron.outgoing) == 1)
    task1 = Task.async(fn -> Neuron.update(input_neuron_pid, %{delta: 0.5}) end)
    Task.await(task1)

    connection_pid = Enum.at(input_neuron.outgoing, 0)
    assert(connection_pid)
    connection = Connection.get(connection_pid)
    assert(connection)
    assert(connection.weight == 0.4)

    task2 = Task.async(fn -> Optimizers.stochastic_gradient_descent(input_neuron) end)
    Task.await(task2)

    assert(connection.weight == 0.4)
  end
end
