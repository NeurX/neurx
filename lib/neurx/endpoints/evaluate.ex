defmodule Neurx.Evaluate do
  @moduledoc """
  Documentation for Train.
  """

  alias Neurx.{Network, Layer, Neuron}

  @doc """
  Evaluates the given data using the network.
  """
  def evaluate(network_pid, data) do
    all_outputs = 
      Enum.map(data, fn sample ->
        # Evaluating.
        network_pid |> Network.get() |> Network.activate(sample.input)

        # Getting the results.
        network = Network.get(network_pid)
        output_layer = Layer.get(network.output_layer)
        outputs = 
          Enum.map(output_layer.neurons, fn neuron_pid ->
            neuron = Neuron.get(neuron_pid)
            neuron.output
          end)
        outputs
      end)
    {:ok, network_pid, all_outputs}
  end
end
