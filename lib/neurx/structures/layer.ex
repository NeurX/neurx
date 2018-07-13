defmodule Neurx.Layer do
  use GenServer
  @moduledoc """
  List of neurons. The are used to apply behaviors on sets of neurons.
  A network is made up layers (which are made up of neurons).
  """

  alias Neurx.{Neuron, Layer}

  defstruct pid: nil, neurons: [], activation_fn: nil, learning_rate: nil, optim_fn: nil

  def start_link(layer_fields \\ %{}) do

    {:ok, pid} = GenServer.start_link(__MODULE__, %Layer{})

    neurons = create_neurons(Map.get(layer_fields, :neuron_size), layer_fields)

    pid |> update(%{pid: pid, neurons: neurons})
    pid |> update(layer_fields)

    {:ok, pid}
  end

  @doc """
  Return a layer by pid.
  """
  def get(pid) do
    GenServer.call(pid, {:get})
  end
  @doc """
  Update a layer by passing in a pid and a map of fields to update.
  """
  def update(pid, fields) do
    # why dont preserve pid
    GenServer.cast(pid, {:update, fields})
  end

  defp create_neurons(nil, nil), do: []
  defp create_neurons(nil, _), do: []
  defp create_neurons(size, _) when size < 1, do: []
  defp create_neurons(size, fields) when size > 0 do
    Enum.into(1..size, [], fn _ ->
      {:ok, pid} = Neuron.start_link(fields)
      pid
    end)
  end

  @doc """
  Add neurons to the layer.
  """
  def add_neurons(layer_pid, neurons) do
    layer_pid |> update(%{neurons: get(layer_pid).neurons ++ neurons})
  end

  @doc """
  Update all deltas for each neuron.
  """
  def train(layer, target_outputs \\ []) do
    layer.neurons
    |> Stream.with_index()
    |> Enum.each(fn tuple ->
      {neuron, index} = tuple
      neuron |> Neuron.train(Enum.at(target_outputs, index))
    end)
  end

  @doc """
  Connect every neuron in the input layer to every neuron in the target layer.
  """
  def connect(input_layer_pid, output_layer_pid) do
    input_layer = get(input_layer_pid)

    unless contains_bias?(input_layer) do
      {:ok, pid} = Neuron.start_link(%{bias?: true, learning_rate: input_layer.learning_rate,
        optim_fn: input_layer.optim_fn})
      input_layer_pid |> add_neurons([pid])
    end

    for source_neuron <- get(input_layer_pid).neurons,
        target_neuron <- get(output_layer_pid).neurons do
      Neuron.connect(source_neuron, target_neuron)
    end
  end

  defp contains_bias?(layer) do
    Enum.any?(layer.neurons, &Neuron.get(&1).bias?)
  end

  @doc """
  Activate all neurons in the layer with a list of values.
  """
  def activate(layer_pid, values \\ nil) do
    layer = get(layer_pid)
    # coerce to [] if nil
    values = List.wrap(values)

    layer.neurons
    |> Stream.with_index()
    |> Enum.each(fn tuple ->
      {neuron, index} = tuple
      neuron |> Neuron.activate(Enum.at(values, index))
    end)
  end
  ## Server Callbacks for GenServer
  def init(layer) do
    {:ok, layer}
  end

  def handle_call({:get}, _from, layer) do
    {:reply, layer, layer}
  end

  def handle_cast({:update, fields}, layer) do
    updated_layer = Map.merge(layer, fields)
    {:noreply, updated_layer}
  end
end
