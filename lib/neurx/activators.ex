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
      #"Relu" -> fn input -> relu(input) end
    end
  end

  @doc """
  Sigmoid function. See more at: https://en.wikipedia.org/wiki/Sigmoid_function
  """
  def sigmoid(input) do
    1 / (1 + :math.exp(-input))
  end

  @doc """
  """
  #def relu(input) do
  #end

end
