defmodule Neurx.LossFunctions do
  @moduledoc """
  Contains all of the loss functions.
  """
  
  alias Neurx.{Neuron, Layer}
  
  @doc """
  Returns the function given its name as a string.
  """
  def retreiveFunction(type) do
    case type do
      "MSE" -> fn n, tar_out -> mseBackPropagation(n, tar_out) end
      nil -> raise "Invalid Loss Function."
    end
  end

  @doc """
  Mean Squared Error backpropagations.
  https://en.wikipedia.org/wiki/Backpropagation#Derivation
  """
  def mseBackPropagation(network, target_outputs) do
    (Layer.get(network.output_layer).neurons
     |> Stream.with_index()
     |> Enum.reduce(0, fn {neuron, index}, sum ->
       target_output = Enum.at(target_outputs, index)
       actual_output = Neuron.get(neuron).output
       squared_error(sum, target_output, actual_output)
     end)) / length(Layer.get(network.output_layer).neurons)
  end

  defp squared_error(sum, target_output, actual_output) do
    sum + 0.5 * :math.pow(target_output - actual_output, 2)
  end
end
