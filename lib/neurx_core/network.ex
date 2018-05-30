defmodule Neurx.Network do
  @moduledoc """
  Contains layers.
  """

  alias Neurx.{Neuron, Layer, Network, Connection}

  defstruct pid: nil, input_layer: nil, hidden_layers: [], output_layer: nil, error: 0

end
