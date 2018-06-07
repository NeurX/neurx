defmodule Neurx.Optimizers do
  @moduledoc """
  Contains all of the optimization functions.
  """

  @doc """
  Returns the function given its name as a string.
  """
  def retreiveFunction(type) do
    case type do
      "SGD" -> nil
      nil -> nil #TODO: set this to an actual default.
    end
  end

end
