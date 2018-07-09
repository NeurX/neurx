defmodule Neurx.Neuron do
  @moduledoc """
  A neuron makes up a network. It's purpose is to sum its inputs
  and compute an output. During training the neurons adjust weights
  of its outgoing connections to other neurons.
  """

  alias Neurx.{Neuron, Connection}

  defstruct pid: nil, input: 0, output: 0, incoming: [], outgoing: [], bias?: false, delta: 0, delta_fn: nil, activation_fn: nil, learning_rate: 0.1, optim_fn: nil

  @doc """
  Create a neuron agent
  """
  def start_link(neuron_fields \\ %{}) do
    {:ok, pid} = Agent.start_link(fn -> %Neuron{} end)
    
    pid |> update(%{pid: pid})
    pid |> update(neuron_fields)

    {:ok, pid}
  end

  @doc """
  ## Pass in the pid, and a map to update values of a neuron
  """
  def update(pid, neuron_fields) do
    Agent.update(pid, &Map.merge(&1, neuron_fields))
  end

  @doc """
  Lookup and return a neuron
  """
  def get(pid), do: Agent.get(pid, & &1)

  @doc """
  Connect two neurons
  """
  def connect(source_neuron_pid, target_neuron_pid) do
    {:ok, connection_pid} = Connection.connection_for(source_neuron_pid, target_neuron_pid)

    source_neuron_pid
    |> update(%{outgoing: get(source_neuron_pid).outgoing ++ [connection_pid]})

    target_neuron_pid
    |> update(%{incoming: get(target_neuron_pid).incoming ++ [connection_pid]})
  end


  defp sumf do
    fn connection_pid, sum ->
      connection = Connection.get(connection_pid)
      sum + get(connection.source_pid).output * connection.weight
    end
  end

  @doc """
  Activate a neuron. Set the input value and compute the output
  Input neuron: output will always equal their input value
  Bias neuron: output is always 1.
  Other neurons: will squash their input value to compute output
  """
  def activate(neuron_pid, value \\ nil) do
    # just to make sure we are not getting a stale agent
    neuron = get(neuron_pid)

    fields =
      if neuron.bias? do
        %{output: 1}
      else
        input = value || Enum.reduce(neuron.incoming, 0, sumf())
        if input_neuron?(neuron) do
          %{input: input, output: input} 
        else
          %{input: input, output: neuron.activation_fn.(input)}
        end
      end

    neuron_pid |> update(fields)
  end

  @doc """
  Backprop using the delta.
  Set the neuron's delta value.
  """
  def train(neuron_pid, target_output \\ nil) do
    # just to make sure we are not getting a stale agent
    neuron = get(neuron_pid)

    if !neuron.bias? && !input_neuron?(neuron) do
      if output_neuron?(neuron) do
        error = (neuron.output - target_output) 
        neuron_pid |> update(%{delta: error * neuron.delta_fn.(neuron.output)})
      else
        neuron |> calculate_outgoing_delta
      end
    end

    neuron.pid |> get |> neuron.optim_fn.()
  end

  defp output_neuron?(neuron) do
    neuron.outgoing == []
  end

  defp input_neuron?(neuron) do
    neuron.incoming == []
  end

  defp calculate_outgoing_delta(neuron) do
    delta =
      Enum.reduce(neuron.outgoing, 0, fn connection_pid, sum ->
        connection = Connection.get(connection_pid)
        sum + connection.weight * get(connection.target_pid).delta
      end)

    neuron.pid |> update(%{delta: delta})
  end
end
