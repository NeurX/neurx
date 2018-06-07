defmodule Neurx.Optimizers do
  @moduledoc """
  Contains all of the optimization functions.
  """

  @doc """
  Returns the function given its name as a string.
  """
  def retreiveFunction(type) do
    case type do
      # TODO: Implement this file.
      "SGD" -> fn n -> nil end
      nil -> raise "Invalid Optimization function."
    end
  end

end
