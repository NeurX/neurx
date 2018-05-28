defmodule NeurxCore.Network do
  @moduledoc """
  Contains layers.
  """

  alias NeurxCore.{Neuron, Layer, Network, Connection}

  defstruct pid: nil, input_layer: nil, hidden_layers: [], output_layer: nil, error: 0

end
