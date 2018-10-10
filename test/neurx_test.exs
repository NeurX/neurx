defmodule NeurxTest do
  use ExUnit.Case
  doctest Neurx

  # alias Neurx.{Network, Layer, Neuron, Activators, LossFunctions, Optimizers}

  ###############################################
  # System Level Network Tests
  ###############################################
  # @tag timeout: :infinity
  # test "Testing network training with large dataset and multiple hidden layers" do
  #   nn = Neurx.build(%{
  #     input_layer: 3,
  #     output_layer: %{
  #       size: 1,
  #       activation: %{
  #         type: "Sigmoid"
  #       }
  #     },
  #     hidden_layers: [
  #       %{
  #         size: 5,
  #         activation: %{
  #           type: "Relu"
  #         }
  #       },
  #       %{
  #         size: 4,
  #         activation: %{
  #           type: "Relu"
  #         }
  #       }
  #     ]
  #   })
  #
  #   assert(nn)
  #
  #   data = TestData.get_haberman_training_data()
  #   options = %{
  #     epochs: 3000,
  #     error_threshold: 0.002,
  #     log_freq: 1
  #   }
  #   {tnn, final_error} = Neurx.train(nn, data, options)
  #
  #   assert(tnn)
  #   assert(final_error)
  #   assert(final_error < 0.01)
  # end

end
