defmodule Neurx.LayerTest do
  use ExUnit.Case
  doctest Neurx.Layer

  alias Neurx.{Neuron, Connection, Layer, Activators}

  test "create new layer with neuron size" do
    {:ok, pid} = Layer.start_link(%{neuron_size: 3})
    layer = Layer.get(pid)
    assert length(layer.neurons) == 3
    assert is_pid(List.first(layer.neurons))
  end

  test "create new layer with empty neuron list when size is negative" do
    {:ok, pid} = Layer.start_link(%{neuron_size: -3})
    layer = Layer.get(pid)
    assert length(layer.neurons) == 0
  end

  test "connect layers: no outgoing or incoming neurons connected" do
    {:ok, input_layer_pid} = Layer.start_link(%{neuron_size: 2})
    {:ok, output_layer_pid} = Layer.start_link(%{neuron_size: 2})

    for neuron <- Layer.get(input_layer_pid).neurons do
      assert length(Neuron.get(neuron).outgoing) == 0
      assert length(Neuron.get(neuron).incoming) == 0
    end

    for neuron <- Layer.get(output_layer_pid).neurons do
      assert length(Neuron.get(neuron).outgoing) == 0
      assert length(Neuron.get(neuron).incoming) == 0
    end
  end

  test "connect layers: input layer's outgoing neurons are stored" do
    {:ok, input_layer_pid} = Layer.start_link(%{neuron_size: 2})
    {:ok, output_layer_pid} = Layer.start_link(%{neuron_size: 2})
    Layer.connect(input_layer_pid, output_layer_pid)

    for neuron <- Layer.get(input_layer_pid).neurons do
      assert length(Neuron.get(neuron).outgoing) == 2
      assert length(Neuron.get(neuron).incoming) == 0

      connection_target_ids =
        Enum.map(Neuron.get(neuron).outgoing, fn connection_pid ->
          Connection.get(connection_pid).target_pid
        end)

      assert connection_target_ids == Layer.get(output_layer_pid).neurons
    end
  end

  test "connect layers: output layer's incoming neurons + bias neuron are stored" do
    {:ok, input_layer_pid} = Layer.start_link(%{neuron_size: 2})
    {:ok, output_layer_pid} = Layer.start_link(%{neuron_size: 2})
    Layer.connect(input_layer_pid, output_layer_pid)

    for neuron <- Layer.get(output_layer_pid).neurons do
      assert length(Neuron.get(neuron).incoming) == 3

      source_neurons =
        Enum.map(Neuron.get(neuron).incoming, fn connection_pid ->
          Neuron.get(Connection.get(connection_pid).source_pid)
        end)

        assert Enum.any?(source_neurons, fn source_neuron -> source_neuron.bias? end)
        assert length(Neuron.get(neuron).outgoing) == 0

        assert Enum.map(Neuron.get(neuron).incoming, fn connection_pid ->
          Connection.get(connection_pid).source_pid
        end) == Layer.get(input_layer_pid).neurons
    end
  end

  test "activate a layer: when values are nil" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pid} = Layer.start_link(%{neuron_size: 2, activation_fn: sigmoid_fn})
    Layer.activate(pid)

    for neuron <- Layer.get(pid).neurons do
      assert Neuron.get(neuron).output == 0
    end
  end

  test "activate a layer: with values" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pid} = Layer.start_link(%{neuron_size: 2, activation_fn: sigmoid_fn})
    pid |> Layer.activate([1, 2])

    for neuron_pid <- Layer.get(pid).neurons do
      neuron = Neuron.get(neuron_pid)
      assert neuron.output == 1 || neuron.output == 2
    end
  end
end
