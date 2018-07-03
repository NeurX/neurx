defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  alias Neurx.{Build}

  @doc """
  Builds the network.
  """
  def build(config) do
    if config == %{} or config == nil do
      raise "[Neurx] :: Configuration is empty or nil."
    end

    {:ok, pid} = Build.build(config)
    pid
  end
  
  @doc """
  Trains the network on given data.
  """
  def train() do
  end
  
  @doc """
  Evaluate the given data using the network.
  """
  def evaluate() do
  end
end
