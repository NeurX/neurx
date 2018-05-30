defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  use Application

  def start(_type, _args) do
    import Supervisor.Spec, warn: true

    children = [
      # Define children/workers and sub-supervisors to be supervised
      # worker(Neurx.Worker, [arg1, arg2, arg3]), etc..
    ]
    opts = [strategy: :one_for_one, name: Neurx.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
