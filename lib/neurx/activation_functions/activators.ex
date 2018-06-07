defmodule Neurx.Activators do
  @moduledoc """
  Contains all of the activations functions.
  """

  @doc """
  Returns the function given its name as a string.
  """
  def retreiveFunction(type) do
    case type do
      "Sigmoid" -> fn input -> sigmoid(input) end
      "Relu" -> fn input -> relu(input) end
      nil -> raise "Invalid Activation function."
    end
  end

  @doc """
  Sigmoid function. See more at: https://en.wikipedia.org/wiki/Sigmoid_function
  """
  def sigmoid(input) do
    1 / (1 + :math.exp(-input))
  end

  @doc """
  Relu function. See more at: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  """
  def relu(input) do
    max(0, input)
  end

end
