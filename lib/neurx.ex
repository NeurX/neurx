defmodule Neurx do
  @moduledoc """
  Documentation for Neurx.
  """

  alias Neurx.{Network}

  @loss_types ["MSE"]
  @default_loss %{type: "MSE"}

  @activation_types ["Sigmoid", "Relu"]
  @default_activation "Sigmoid"

  @optim_types ["SGD"]
  @default_optim %{type: "SGD", learning_rate: 0.1}
  @default_learning_rate 0.1

  @doc """
  Builds the network.
  """
  def build(config) do
    if config[:input_layer] <= 0 do raise "Invalid size of input layer." end

    # Sanitize the output layer config.
    output_layer =
      if config[:output_layer] != nil do
        if config[:output_layer][:size] <= 0 do
          raise "Invalid size of output layer."
        end
        %{size: config[:output_layer][:size],
         activation: sanitize_activation_functions(config[:output_layer][:activation])}
      end

    # Sanitize hidden layer configs.
    hidden_layers = 
      if config[:hidden_layers] != nil do
        Enum.map(config[:hidden_layers], fn hl ->
          size = hl[:size]
          if size != nil do
            if size <= 0 do raise "Invalid size of hidden layer." end
            activation = sanitize_activation_functions(hl[:activation])
            prefuncs = sanitize_function_list(hl[:prefix_functions])
            suffuncs = sanitize_function_list(hl[:suffix_functions])
            %{size: size, activation: activation, prefix_functions: prefuncs, 
              suffix_functions: suffuncs}
            end
          end)
      end

    # Sanitize loss function config.
    loss = 
      if config[:loss_function] != nil do
        if config[:loss_function][:type] == nil do raise "Loss type cannot be null." end
        if config[:loss_function][:type] not in @loss_types do raise "Unknown loss type." end
          %{type: config[:loss_function][:type]}
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
            %{type: otype, learning_rate: config[:optim_functions][:learning_rate]}
          else
            %{type: otype, learning_rate: @default_learning_rate}
        end
      else
        @default_optim
      end

    # Recreating the config.
    sanitized_config = %{
      input_layer: config[:input_layer],
      output_layer: output_layer,
      hidden_layers: hidden_layers,
      loss_function: loss,
      optimization_function: optim
    }

    {:ok, pid} = Network.start_link(sanitized_config)
    pid
  end

  defp sanitize_activation_functions(activ) do
    if activ != nil do
      if activ[:type] not in @activation_types do
        raise "Invalid activation function."
      else
         activ[:type]
      end
    else
      @default_activation
    end
  end

  defp sanitize_function_list(funcs) do
    if funcs != nil do
      funcs |>
      Enum.map(fn f ->
        if f != nil do
          f
        end
      end)
    else
      []
    end
  end
  
  @doc """
  Trains the network on the given data.
  """
  def train(network_pid, data, options \\ %{}) do
    epochs = options.epochs
    log_freqs = options.log_freqs
    data_length = length(data)

    for epoch <- 0..epochs do
      average_error =
        Enum.reduce(data, 0, fn sample, sum ->
          # sum weighted inputs to produce output value of network
          # that output will be compared with target output to find the delta
          network_pid |> Network.get() |> Network.activate(sample.input)

          # Backpropagation
          network_pid |> Network.get() |> Network.train(sample.output)

          sum + Network.get(network_pid).error / data_length
        end)

      if rem(epoch, log_freqs) == 0 || epoch + 1 == epochs do
        IO.puts("Epoch: #{epoch}   Error: #{unexponential(average_error)}")
      end
    end
  end

  defp unexponential(average_error) do
    :erlang.float_to_binary(average_error, [{:decimals, 19}, :compact])
  end
end
