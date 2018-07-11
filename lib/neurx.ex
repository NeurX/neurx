defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  alias Neurx.{Build, Train}

  @doc """
  Builds the network.

  Returns PID of network built.
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

  Returns PID of network trained
  """
  def train(network_pid, training_data, options \\ %{}) do
    pid =
      case {network_pid, training_data} do
        {nil, nil} ->
          raise "[Neurx] :: Invalid network PID and training data."
        {_, nil} ->
          raise "[Neurx] :: Invalid training data."
        {nil, _} ->
          raise "[Neurx] :: Invalid network PID."
        _ ->
          {:ok, pid} = Train.train(network_pid, training_data, options)
          pid
      end
    pid
  end

  @doc """
  Evaluate the given data using the network.
  """
  def evaluate() do
  end

  @doc """
  Export a built network.
  """
  def export() do
  end

  @doc """
  Import a built network file.
  """
  def import() do
  end
end
