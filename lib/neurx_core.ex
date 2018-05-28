defmodule NeurxCore do
  @moduledoc """
  Documentation for NeurxCore.
  """

  use Application

  def start(_type, _args) do
    import Supervisor.Spec, warn: true

    children = [
      # Define children/workers and sub-supervisors to be supervised
      # worker(NeurxCore.Worker, [arg1, arg2, arg3]), etc..
    ]
    opts = [strategy: :one_for_one, name: NeurxCore.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
