defmodule Neurx.Connection do
  use GenServer

  @moduledoc """
  Neurons communciate via connections.
  Connection weights determine the network output and are updated while training occurs.
  Network capability is represented in the network matrix of weight values.
  """

  alias Neurx.{Connection}

  defstruct pid: nil, source_pid: nil, target_pid: nil, weight: 0.4

  @doc """
  Creates a connection.
  """
  def start_link(connection_fields \\ %{}) do
    {:ok, pid} = GenServer.start_link(__MODULE__, %Connection{})
    update(pid, Map.merge(connection_fields, %{pid: pid}))

    {:ok, pid}
  end

  @doc """
  Return connection by PID
  """
  def get(pid) do
    GenServer.call(pid, {:get})
  end

  @doc """
  Update a connection by passing in a pid and a map of fields to update.
  """
  def update(pid, fields) do
    GenServer.cast(pid, {:update, fields})
  end

  @doc """
  Convenience method that takes two neuron pids, creates a connection,
  and then returns connection pid.
  """
  def connection_for(source_pid, target_pid) do
    {:ok, pid} = start_link()
    pid |> update(%{source_pid: source_pid, target_pid: target_pid})

    {:ok, pid}
  end

  @doc """
  Server Callback for GenServer - init
  """
  def init(connection) do
    {:ok, connection}
  end

  @doc """
  Server Callback for GenServer - call
  """
  def handle_call({:get}, _from, connection) do
    {:reply, connection, connection}
  end

  @doc """
  Server Callback for GenServer - cast
  """
  def handle_cast({:update, fields}, connection) do
    updated_connection = Map.merge(connection, fields)
    {:noreply, updated_connection}
  end
end
