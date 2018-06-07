defmodule Neurx.Network do
  @moduledoc """
  Contains layers which makes up a matrix of neurons.
  """

  alias Neurx.{Neuron, Layer, Network, Activators, LossFunctions, Optimizers}

  defstruct pid: nil, input_layer: nil, hidden_layers: [], output_layer: nil, error: 0, loss_fn: nil, optim_fn: nil

  @doc """
  Pass in layer sizes which will generate the layers for the network.
  The first number represents the number of neurons in the input layer.
  The last number represents the number of neurons in the output layer.
  [Optionally] The middle numbers represent the number of neurons for hidden layers.
  """
  def start_link(config) do
    {:ok, pid} = Agent.start_link(fn -> %Network{} end)

    layers =
      map_layers(
        input_neurons(Map.get(config, :input_layer)),
        hidden_neurons(Map.get(config, :hidden_layers)),
        output_neurons(Map.get(config, :output_layer))
      )

    pid |> update(layers)
    pid |> update(%{optim_fn: Optimizers.retreiveFunction(
      Map.get(Map.get(config, :optimization_function), :type)), 
      loss_fn: LossFunctions.retreiveFunction(Map.get(Map.get(config, :loss_function), :type))})
    pid |> connect_layers

    {:ok, pid}
  end

  @doc """
  Return the network by pid.
  """
  def get(pid), do: Agent.get(pid, & &1)

  @doc """
  Update the network layers.
  """
  def update(pid, fields) do
    # preserve the pid!!
    fields = Map.merge(fields, %{pid: pid})
    Agent.update(pid, &Map.merge(&1, fields))
  end

  defp input_neurons(size) do
    {:ok, pid} = Layer.start_link(%{neuron_size: size})
    pid
  end

  defp hidden_neurons(hidden_layers) do
    if hidden_layers != nil do
      hidden_layers
      |> Enum.map(fn layer ->
        size = Map.get(layer, :size)
          {:ok, pid} = Layer.start_link(%{neuron_size: size,
            activation_fn: Activators.retreiveFunction(Map.get(layer, :activation))})
        pid
      end)
    else
      []
    end
  end

  defp output_neurons(layer_fields) do
    size = Map.get(layer_fields, :size)
    {:ok, pid} = Layer.start_link(%{neuron_size: Map.get(layer_fields, :size),
      activation_fn: Activators.retreiveFunction(Map.get(layer_fields, :activation))})
    pid
  end

  defp connect_layers(pid) do
    layers = pid |> Network.get() |> flatten_layers
    
    layers
    |> Stream.with_index()
    |> Enum.each(fn tuple ->
      {layer, index} = tuple
      next_index = index + 1

      if Enum.at(layers, next_index) do
        Layer.connect(layer, Enum.at(layers, next_index))
      end
    end)
  end

  defp flatten_layers(network) do
    if network.hidden_layers != nil do
      [network.input_layer] ++ network.hidden_layers ++ [network.output_layer]
    else
      [network.input_layer] ++ [network.output_layer]
    end
  end

  @doc """
  Activate the network given list of input values.
  """
  def activate(network, input_values) do
    network.input_layer |> Layer.activate(input_values)

    Enum.map(network.hidden_layers, fn hidden_layer ->
      hidden_layer |> Layer.activate()
    end)

    network.output_layer |> Layer.activate()
  end

  @doc """
  Set the network error and output layer's deltas propagate them
  backward through the network. (Back Propogation!)

  The input layer is skipped (no use for deltas).
  """
  def train(network, target_outputs) do
    network.output_layer |> Layer.get() |> Layer.train(target_outputs)
    network.pid |> update(%{error: network.loss_function(network, target_outputs)})

    network.hidden_layers
    |> Enum.reverse()
    |> Enum.each(fn layer_pid ->
      Layer.get(layer_pid) |> Layer.train(target_outputs)
    end)

    network.input_layer |> Layer.get() |> Layer.train(target_outputs)
  end

  defp map_layers(input_layer, hidden_layers, output_layer) do
    %{
      input_layer: input_layer,
      output_layer: output_layer,
      hidden_layers: hidden_layers
    }
  end
end
