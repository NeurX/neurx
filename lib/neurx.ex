defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  use Application
  require Neurx.Network

  @doc """
  def start(_type, _args) do
    import Supervisor.Spec, warn: true

    children = [
      # Define children/workers and sub-supervisors to be supervised
      # worker(Neurx.Worker, [arg1, arg2, arg3]), etc..
    ]
    opts = [strategy: :one_for_one, name: Neurx.Supervisor]
    Supervisor.start_link(children, opts)
  end
  """

  @loss_types ["MSE"]
  @default_loss [type: "MSE"]

  @optim_types ["SGD"]
  @default_optim [type: "SGD", learning_rate: 0.1]
  @default_learning_rate 0.1

  def build(config) do
    if config[:input_layer] <= 0 do raise "Invalid size of input layer." end
    if config[:output_layer] <= 0 do raise "Invalid size of output layer." end

    # Sanitize hidden layer configs.
    hidden_layers = []
    if config[:hidden_layers] != nil do
      Enum.each config[:hidden_layers], fn hl ->
        size = hl[:size]
        if size != nil do
          if size <= 0 do raise "Invalid size of hidden layer." end
          prefuncs = sanitize_function_list(hl[:prefix_functions])
          suffuncs = sanitize_function_list(hl[:suffix_functions])
          hidden_layers = hidden_layers ++ [size: size, prefix_functions: prefuncs, suffix_functions: suffuncs]
        end
      end
    end

    # Sanitize loss function config.
    loss = 
      if config[:loss_function] != nil do
        if config[:loss_function][:type] == nil do raise "Loss type cannot be null." end
        if config[:loss_function][:type] not in @loss_types do raise "Unknown loss type." end
          [type: config[:loss_function][:type]]
      else
        @default_loss
      end

      # Sanitize optim function config.
    optim =
      if config[:optim_function] != nil do
        if config[:optim_function][:type] == nil do raise "Optimization type cannot be null." end
        if config[:optim_function][:type] not in @optim_types do raise "Unknown optimization type." end
          otype = config[:optim_function][:type]
          if config[:optim_functions][:learning_rate] > 0 do
            [type: otype, learning_rate: config[:optim_functions][:learning_rate]]
          else
            [type: otype, learning_rate: @default_learning_rate]
        end
      else
        @default_optim
      end

    # Recreating the config.
    sanitized_config = %{
      input_layer: config[:input_layer],
      output_layer: config[:output_layer],
      hidden_layers: hidden_layers,
      loss_function: loss,
      optimization_function: optim
    }

    #{:ok, pid} = Network.create(sanitized_config)
    #pid
    nil
  end

  defp sanitize_function_list(funcs) do
    functions = nil
    if funcs != nil do
      Enum.each funcs, fn f ->
        if f != nil do
          if functions == nil do
            functions = [f]
          else
            functions = [functions | f]
          end
        end
      end
    end
    functions
  end
end
