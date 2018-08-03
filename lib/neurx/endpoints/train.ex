defmodule Neurx.Train do
  @moduledoc """
  Documentation for Train.
  """

  alias Neurx.{Network}

  @default_error 0.01
  @default_log_freq 100

  @doc """
  Trains the network.
  """
  def train(network_pid, training_data, options) do
    filtered_options =
      case options do
        nil ->
          %{epochs: nil, error_threshold: @default_error, log_freq: @default_log_freq}
        %{} ->
          epochs =
            Map.get(options, :epochs, -1)
            |> (fn(x) -> if x <= 0, do: nil, else: x end).()

          error_threshold =
            Map.get(options, :error_threshold, -1)
            |> (fn(x) -> if x <= 0, do: nil, else: x end).()

          log_freq =
            Map.get(options, :log_freq, @default_log_freq)
            |>(fn(x) -> if x < 0, do: @default_log_freq, else: x end).()
          %{epochs: epochs, error_threshold: error_threshold, log_freq: log_freq}
        _ ->
          %{epochs: nil, error_threshold: @default_error, log_freq: @default_log_freq}
      end

    if (filtered_options.log_freq > 0), do: IO.puts("\n")

    do_training(network_pid, training_data, filtered_options, length(training_data), 0, 1)
  end

  defp do_training(network_pid, training_data, options, num_training_samples, epoch, last_error) do
    # IO.inspect(options)
    # IO.inspect(options.error_threshold)

    average_error =
      Enum.reduce(training_data, 0,
        fn sample, sum ->
          network_pid |> Network.get() |> Network.activate(sample.input)
          network_pid |> Network.get() |> Network.train(sample.output)
          sum + Network.get(network_pid).error / num_training_samples
      end)

    if options.log_freq != 0 && (rem(epoch, options.log_freq) == 0 || epoch + 1 == options.epochs) do
      IO.puts("Epoch: #{epoch} \t\tError: #{unexponential(average_error)} \t\tDelta: #{unexponential(last_error-average_error)}")
    end

    if (options.log_freq > 0) do

    end
    cond do
      (options.error_threshold != nil) and (average_error <= options.error_threshold) ->
        if (options.log_freq > 0) do
          IO.inspect("END on ERROR")
          IO.puts("Epoch: #{epoch}    Error: #{unexponential(average_error)}")
        end
        {:ok, network_pid, Network.get(network_pid).error}
      (options.epochs != nil) and (epoch >= options.epochs) ->
        if (options.log_freq > 0) do
          IO.inspect("END on EPOCHS")
          IO.puts("Epoch: #{epoch}    Error: #{unexponential(average_error)}")
        end
        {:ok, network_pid, Network.get(network_pid).error}
      true ->
        do_training(network_pid, training_data, options, num_training_samples, epoch+1, average_error)
    end
  end


  defp unexponential(average_error) do
    :erlang.float_to_binary(average_error, [{:decimals, 19}, :compact])
  end
end
