defmodule NeurxCore.Connection do
  @moduledoc """
  Neurons communciate via connections.
  """
  alias NeurxCore.{Connection}

  defstruct pid: nil, source_pid: nil, target_pid: nil, weight: 0.4

  def start_link(connection_fields \\ %{}) do
    {:ok}
  end
end
