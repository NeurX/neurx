defmodule Neurx.ActivatorsTest do
  use ExUnit.Case
  doctest Neurx.Activators

  alias Neurx.{Activators}
  
  test "Testing function retrieval." do
    assert(Activators.retreiveFunction("Sigmoid"))
    assert(Activators.retreiveFunction("Relu"))
    try do
      assert(Activators.retreiveFunction("REKT"))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
    try do
      assert(Activators.retreiveFunction(nil))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Testing Sigmoid function." do
    assert(Activators.sigmoid(1) == 0.7310585786300049)
  end

  test "Testing Relu function." do
    assert(Activators.relu(1) == 1)
    assert(Activators.relu(0) == 0)
    assert(Activators.relu(-1) == 0)
  end
end
