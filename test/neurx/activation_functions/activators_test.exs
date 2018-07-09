defmodule Neurx.ActivatorsTest do
  use ExUnit.Case
  doctest Neurx.Activators

  alias Neurx.{Activators}
  
  test "Testing function retrieval." do
    assert(Activators.getFunction("Sigmoid"))
    assert(Activators.getFunction("Relu"))
    try do
      assert(Activators.getFunction("REKT"))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
    try do
      assert(Activators.getFunction(nil))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Testing delta function retrieval." do
    assert(Activators.getDeltaFunction("Sigmoid"))
    assert(Activators.getDeltaFunction("Relu"))
    try do
      assert(Activators.getDeltaFunction("REKT"))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
    try do
      assert(Activators.getDeltaFunction(nil))
      assert(false) # Should not get here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Testing Sigmoid function." do
    assert(Activators.sigmoid(1) == 0.7310585786300049)
  end
  
  test "Testing Sigmoid function overflow." do
    # If overflow is not caught Elixir will crash since
    # e^720 is larger then the max float.
    assert(Activators.sigmoid(-720) == 6.643397797997951e-307)
  end

  test "Testing Relu function." do
    assert(Activators.relu(1) == 1)
    assert(Activators.relu(0) == 0)
    assert(Activators.relu(-1) == 0)
  end
  
  test "Testing Sigmoid derivative function." do
    assert(Activators.sigmoid_derivative(1) == 0.19661193324148185)
  end

  test "Testing Relu derivative function." do
    assert(Activators.relu_derivative(1) == 1)
    assert(Activators.relu_derivative(0) == 1) # Function is technically undefined for zero.
    assert(Activators.relu_derivative(-1) == 0)
  end
end
