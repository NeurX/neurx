defmodule Neurx.Activators do
  @moduledoc """
  Contains all of the activations functions.
  """

  @doc """
  Returns the function given its name as a string.
  """
  def getFunction(type) do
    cond do
      type == "Sigmoid" -> fn input -> sigmoid(input) end
      type == "Relu" -> fn input -> relu(input) end
      !String.valid?(type) and type[:custom] != nil -> Enum.at(type[:custom], 0)
      type == nil -> raise "[Neurx.Activators] :: Invalid Activation Function."
      true -> raise "[Neurx.Activators] :: Unknown Activation Function."
    end
  end

  def getDeltaFunction(type) do
    cond do
      type == "Sigmoid" -> fn input -> sigmoid_derivative(input) end
      type == "Relu" -> fn input -> relu_derivative(input) end
      !String.valid?(type) and type[:custom] != nil -> Enum.at(type[:custom], 1)
      type == nil -> raise "[Neurx.Activators] :: Invalid Delta Function."
      true -> raise "[Neurx.Activators] :: Unknown Delta Function."
    end
  end

  @doc """
  Sigmoid function. See more at: https://en.wikipedia.org/wiki/Sigmoid_function
  """
  def sigmoid(input) do
    if -input > 705 do
      1 / (1 + :math.exp(705))
    else
      1 / (1 + :math.exp(-input))
    end
  end

  @doc """
  Derivative of the sigmoid function for back propagation.
  """
  def sigmoid_derivative(input) do
    sig = sigmoid(input)
    sig * (1 - sig)
  end

  @doc """
  Relu function. See more at: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  """
  def relu(input) do
    max(0, input)
  end

  @doc """
  Derivative of the relu function for back propagation.
  """
  def relu_derivative(input) do
    if input < 0 do
      0
    else
      1
    end
  end

end
