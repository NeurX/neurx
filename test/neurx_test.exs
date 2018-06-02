defmodule NeurxTest do
  use ExUnit.Case
  doctest Neurx

  require Neurx

  test "Create basic network." do
    config = %{
      input_layer: 3,
      output_layer: 1
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end
  
  test "Create empty network." do
    nn = Neurx.build(%{})
    assert(nn == nil)
  end
  
  test "Create network with invalid input sizes." do
    config1 = %{
      input_layer: -3,
      output_layer: 1
    }

    config2 = %{
      input_layer: "hello",
      output_layer: 1
    }

    nn1 = Neurx.build(config1)
    nn2 = Neurx.build(config2)
    assert(nn1 == nil)
    assert(nn2 == nil)
  end
  
  test "Create network with specific loss function." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      loss_function: %{
        type: "MSE"
      }
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end
  
  test "Create network with unknown loss function." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      loss_function: %{
        type: "Not Hotdog"
      }
    }

    nn = Neurx.build(config)
    assert(nn == nil)
  end

  test "Create network with specific optimization function." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      optim_function: %{
        type: "SGD",
        learning_rate: 0.9
      }
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end
  
  test "Create network with unknown optimization function." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      optim_function: %{
        type: "Hotdog",
        learning_rate: 0.9
      }
    }

    nn = Neurx.build(config)
    assert(nn == nil)
  end
  
  test "Create network with invalid learning rate." do
    config1 = %{
      input_layer: 3,
      output_layer: 1,
      optim_function: %{
        type: "SGD",
        learning_rate: -0.1
      }
    }
    config2 = %{
      input_layer: 3,
      output_layer: 1,
      optim_function: %{
        type: "SGD",
        learning_rate: "SGD"
      }
    }

    nn1 = Neurx.build(config1)
    nn2 = Neurx.build(config2)
    assert(nn1 == nil)
    assert(nn2 == nil)
  end

  test "Create Network with one hidden layer." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      hidden_layers: [
        %{
          size: 2
        }
      ]
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end
  
  test "Create Network with a hidden layer with invalid size." do
    config1 = %{
      input_layer: 3,
      output_layer: 1,
      hidden_layers: [
        %{
          size: -50
        }
      ]
    }
    config2 = %{
      input_layer: 3,
      output_layer: 1,
      hidden_layers: [
        %{
          size: "DurpLearing"
        }
      ]
    }

    nn1 = Neurx.build(config1)
    nn2 = Neurx.build(config2)
    assert(nn1 == nil)
    assert(nn2 == nil)
  end

  test "Create Network with two hidden layer." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      hidden_layers: [
        %{
          size: 2
        },
        %{
          size: 3
        }
      ]
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end

  test "Create Network with one hidden layer with prefix & suffix functions." do
    config = %{
      input_layer: 3,
      output_layer: 1,
      hidden_layers: [
        %{
          size: 2,
          prefix_functions: [
            &(&1 * 0.1)
          ],
          suffix_functions: [
            &(&1 / 0.2)
          ]
        }
      ]
    }

    nn = Neurx.build(config)
    assert(nn != nil)
  end

end
