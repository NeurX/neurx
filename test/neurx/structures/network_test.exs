defmodule Neurx.NetworkTest do
  use ExUnit.Case
  doctest Neurx.Network

  alias Neurx.{Layer, Network}

  test "keep track of error with default" do
    pid = Neurx.build(%{
      input_layer: 2,
      output_layer: %{
        size: 1
      }
    })
    assert Network.get(pid).error == 0
  end
end
