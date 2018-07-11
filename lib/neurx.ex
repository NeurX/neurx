defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  alias Neurx.{Build, Train, Evaluate}

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
    {:ok} = verify_pid_data_tuple(network_pid, training_data)
    {:ok, pid, final_error} = Train.train(network_pid, training_data, options)
    {pid, final_error}
  end

  @doc """
  Evaluate the given data using the network.
  """
  def evaluate(network_pid, data) do
    {:ok} = verify_pid_data_tuple(network_pid, data)
    {:ok, pid, outputs} = Evaluate.evaluate(network_pid, data)
    {pid, outputs}
  end

  defp verify_pid_data_tuple(pid, data) do
    status = 
      case {pid, data} do
        {nil, nil} ->
          raise "[Neurx] :: Invalid network PID and data."
        {_, nil} ->
          raise "[Neurx] :: Invalid data."
        {nil, _} ->
          raise "[Neurx] :: Invalid network PID."
        _ ->
          {:ok}
      end
    status
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
