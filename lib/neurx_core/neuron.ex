defmodule NeurxCore.Neuron do
  @moduledoc """
  A neuron makes up a network.
  """

  alias NeurxCore.{Neuron, Connection}

  defstruct pid: nil, input: 0, output: 0, incoming: [], outgoing: [], bias?: false, delta: 0

end
