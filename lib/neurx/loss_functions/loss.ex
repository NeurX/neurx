defmodule Neurx.LossFunctions do
  @moduledoc """
  Contains all of the loss functions.
  """
  
  alias Neurx.{Neuron}
  
  @doc """
  Returns the function given its name as a string.
  """
  def getFunction(type) do
    case type do
      "MSE" -> fn act_out, tar_out -> mean_squared_error(act_out, tar_out) end
      nil -> raise "[Neurx.LossFunctions] :: Invalid Loss Function."
      _ -> raise "[Neurx.LossFunctions] :: Unknown Loss Function."
    end
  end

  @doc """
  Mean Squared Error.
  """
  def mean_squared_error(actual_outputs, target_outputs) do
    (actual_outputs
     |> Stream.with_index()
     |> Enum.reduce(0, fn {neuron, index}, sum ->
       target_output = Enum.at(target_outputs, index)
       actual_output = Neuron.get(neuron).output
       squared_error(sum, target_output, actual_output)
     end)) / length(actual_outputs)
  end

  defp squared_error(sum, target_output, actual_output) do
    sum + :math.pow(target_output - actual_output, 2)
  end
end
