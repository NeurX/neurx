defmodule Neurx.Optimizers do
  @moduledoc """
  Contains all of the optimization functions.
  """

  alias Neurx.{Connection, Neuron}

  @doc """
  Returns the function given its name as a string.
  """
  def getFunction(type) do
    case type do
      "SGD" -> fn neuron -> stochastic_gradient_descent(neuron) end
      nil -> raise "[Neurx.Optimizers] :: Invalid Optimization Function."
      _ -> raise "[Neurx.Optimizers] :: Unknown Optimization Function."
    end
  end


  @doc """
  Performs a stochastic gradient descent step on a
  single neuron's connections.
  https://en.wikipedia.org/wiki/Stochastic_gradient_descent
  """
  def stochastic_gradient_descent(neuron) do
    for connection_pid <- neuron.outgoing do
      connection = Connection.get(connection_pid)
      gradient = neuron.output * Neuron.get(connection.target_pid).delta
      updated_weight = connection.weight - gradient * neuron.learning_rate
      Connection.update(connection_pid, %{weight: updated_weight})
    end
  end
end
