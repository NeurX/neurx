defmodule Neurx.Train do
  @moduledoc """
  Documentation for Train.
  """

  alias Neurx.{Network}

  @default_epoch 0
  @default_log_freq 0

  @doc """
  Trains the network.
  """
  def train(network_pid, training_data, options) do
    {epochs, log_freq} =
      case options do
        nil ->
          {@default_epoch, @default_log_freq}
        %{} ->
          epochs = 
            case options.epochs do
              nil ->
                @default_epoch
              x when x <= 0 ->
                @default_epoch
              _ ->
                options.epochs
            end
          log_freq = 
            case options.log_freq do
              nil ->
                @default_log_freq
              x when x <= 0 ->
                @default_log_freq
              _ ->
                options.log_freq
            end
          {epochs, log_freq}
        _ ->
          {@default_epoch, @default_log_freq}
      end

    num_training_samples = length(training_data)
    IO.puts "\n"

    for epoch <- 0..epochs do
      average_error =
        Enum.reduce(training_data, 0, fn sample, sum ->
          network_pid |> Network.get() |> Network.activate(sample.input)
          network_pid |> Network.get() |> Network.train(sample.output)

          sum + Network.get(network_pid).error / num_training_samples
        end)

      if log_freq != 0 && (rem(epoch, log_freq) == 0 || epoch + 1 == epochs) do
        IO.puts("Epoch: #{epoch}    Error: #{unexponential(average_error)}")
      end
    end

    {:ok, network_pid, Network.get(network_pid).error}
  end

  defp unexponential(average_error) do
    :erlang.float_to_binary(average_error, [{:decimals, 19}, :compact])
  end
end
