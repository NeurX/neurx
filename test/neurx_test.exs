defmodule NeurxTest do
  use ExUnit.Case
  doctest Neurx

  alias Neurx.{Network, Layer, Neuron, Activators, LossFunctions, Optimizers}
  
  ###############################################
  # Network Building Tests
  ###############################################
  
  test "Create all default 2:1 network." do
    nn = Neurx.build(%{
      input_layer: 2,
      output_layer: %{
        size: 1
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.output_layer)
    assert(network.hidden_layers == [])
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)

    sigmoid = Activators.getFunction("Sigmoid")
    sigmoid_derv = Activators.getDeltaFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(sigmoid)
    assert(sigmoid_derv)
    assert(mse)
    assert(sgd)
    
    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 3)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.delta_fn == nil)
      assert(neuron.learning_rate == 0.1)
      assert(neuron.optim_fn == sgd)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.delta_fn == sigmoid_derv)
      assert(neuron.learning_rate == 0.1)
      assert(neuron.optim_fn == sgd)
    end)
  end

  test "Create basic 3:1 network." do
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1,
        activation: %{
          type: "Sigmoid"
        }
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.output_layer)
    assert(network.hidden_layers == [])
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)

    sigmoid = Activators.getFunction("Sigmoid")
    sigmoid_derv = Activators.getDeltaFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(sigmoid)
    assert(sigmoid_derv)
    assert(mse)
    assert(sgd)
    
    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.delta_fn == nil)
      assert(neuron.learning_rate == 0.1)
      assert(neuron.optim_fn == sgd)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.delta_fn == sigmoid_derv)
      assert(neuron.learning_rate == 0.1)
      assert(neuron.optim_fn == sgd)
    end)
  end
  
  test "Create basic 3:2:1 network." do
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1,
        activation: %{
          type: "Sigmoid"
        }
      },
      hidden_layers: [
        %{
          size: 2
        }
      ],
      loss_function: %{
        type: "MSE"
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.hidden_layers)
    assert(length(network.hidden_layers) == 1)
    assert(network.output_layer)
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)
    
    sigmoid = Activators.getFunction("Sigmoid")
    sigmoid_derv = Activators.getDeltaFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(sigmoid)
    assert(sigmoid_derv)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)
    
    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.delta_fn == nil)
      assert(neuron.learning_rate == 0.1)
    end)

    Enum.each(network.hidden_layers, fn lpid ->
      layer = Layer.get(lpid)
      assert(layer)
      assert(length(layer.neurons) == 3)
      assert(layer.activation_fn == sigmoid)
      Enum.each(layer.neurons, fn npid ->
        neuron = Neuron.get(npid)
        assert(neuron)
        assert(neuron.learning_rate == 0.1)
        assert(neuron.optim_fn == sgd)
        assert(neuron.delta_fn == nil)
        if neuron.bias? do
          assert(neuron.activation_fn == nil)
        else
          assert(neuron.activation_fn == sigmoid)
        end
      end)
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.delta_fn == sigmoid_derv)
      assert(neuron.learning_rate == 0.1)
    end)
  end
  
  test "Create 100:50:25:2 network." do
    nn = Neurx.build(%{
      input_layer: 100,
      output_layer: %{
        size: 2
      },
      hidden_layers: [
        %{
          size: 50,
          activation: %{
            type: "Relu"
          }
        },
        %{
          size: 25,
          activation: %{
            type: "Relu"
          }
        }
      ],
      loss_function: %{
        type: "MSE"
      },
      optim_function: %{
        type: "SGD",
        learning_rate: 0.3
      }
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.hidden_layers)
    assert(length(network.hidden_layers) == 2)
    assert(network.output_layer)
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)
    
    relu = Activators.getFunction("Relu")
    relu_derv = Activators.getDeltaFunction("Relu")
    sigmoid = Activators.getFunction("Sigmoid")
    sigmoid_derv = Activators.getDeltaFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(relu)
    assert(relu_derv)
    assert(sigmoid)
    assert(sigmoid_derv)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 101)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)
    
    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.delta_fn == nil)
      assert(neuron.learning_rate == 0.3)
    end)

    hl1 = Enum.at(network.hidden_layers, 0)
    hlayer1 = Layer.get(hl1)
    assert(hlayer1)
    assert(length(hlayer1.neurons) == 51)
    assert(hlayer1.activation_fn == relu)
    Enum.each(hlayer1.neurons, fn npid ->
      neuron = Neuron.get(npid)
      assert(neuron)
      assert(neuron.learning_rate == 0.3)
      assert(neuron.optim_fn == sgd)
      assert(neuron.delta_fn == nil)
      if neuron.bias? do
        assert(neuron.activation_fn == nil)
      else
        assert(neuron.activation_fn == relu)
      end
    end)

    hl2 = Enum.at(network.hidden_layers, 1)
    hlayer2 = Layer.get(hl2)
    assert(hlayer2)
    assert(length(hlayer2.neurons) == 26)
    assert(hlayer2.activation_fn == relu)
    Enum.each(hlayer2.neurons, fn npid ->
      neuron = Neuron.get(npid)
      assert(neuron)
      assert(neuron.learning_rate == 0.3)
      assert(neuron.optim_fn == sgd)
      assert(neuron.delta_fn == nil)
      if neuron.bias? do
        assert(neuron.activation_fn == nil)
      else
        assert(neuron.activation_fn == relu)
      end
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 2)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.delta_fn == sigmoid_derv)
      assert(neuron.learning_rate == 0.3)
    end)
  end

  # TODO: prefix and suffix functions don't do anything right now.
  # Once they actually do something we will need to add some asserts
  # to make sure they are where they are supposed to be.
  test "Create network with hidden layer containing prefix & suffix functions." do
    pref = fn n -> n * 0.1 end
    suff = fn n -> n / 0.2 end
    nn = Neurx.build(%{
      input_layer: 3,
      output_layer: %{
        size: 1
      },
      hidden_layers: [
        %{
          size: 2,
          prefix_functions: [
            pref
          ],
          suffix_functions: [
            suff
          ]
        }
      ]
    })

    assert(nn)

    network = Network.get(nn)

    assert(network)
    assert(network.input_layer)
    assert(network.hidden_layers)
    assert(length(network.hidden_layers) == 1)
    assert(network.output_layer)
    assert(network.loss_fn)

    input_layer = Layer.get(network.input_layer)
    output_layer = Layer.get(network.output_layer)
    
    sigmoid = Activators.getFunction("Sigmoid")
    mse = LossFunctions.getFunction("MSE")
    sgd = Optimizers.getFunction("SGD")
    assert(sigmoid)
    assert(mse)
    assert(sgd)

    assert(network.loss_fn == mse)
    assert(network.optim_fn == sgd)

    assert(input_layer)
    assert(length(input_layer.neurons) == 4)
    assert(input_layer.activation_fn == nil)
    assert(input_layer.optim_fn == sgd)

    Enum.each(network.hidden_layers, fn lpid ->
      layer = Layer.get(lpid)
      assert(layer)
      assert(length(layer.neurons) == 3)
      assert(layer.activation_fn == sigmoid)
      Enum.each(layer.neurons, fn npid ->
        neuron = Neuron.get(npid)
        assert(neuron)
        assert(neuron.learning_rate == 0.1)
        assert(neuron.optim_fn == sgd)
        if neuron.bias? do
          assert(neuron.activation_fn == nil)
        else
          assert(neuron.activation_fn == sigmoid)
        end
      end)
    end)
    
    assert(output_layer)
    assert(length(output_layer.neurons) == 1)
    assert(output_layer.activation_fn == sigmoid)
    assert(output_layer.optim_fn == sgd)

    Enum.each(input_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == nil)
      assert(neuron.learning_rate == 0.1)
    end)
    
    Enum.each(output_layer.neurons, fn pid ->
      neuron = Neuron.get(pid)
      assert(neuron)
      assert(neuron.activation_fn == sigmoid)
      assert(neuron.learning_rate == 0.1)
    end)
  end

  ###############################################
  # Invalid Config Tests
  ###############################################

  test "Empty build config." do
    try do
      Neurx.build(%{})
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Nil build config." do
    try do
      Neurx.build(nil)
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end

  test "Invalid input size." do
    try do
      Neurx.build(%{
        input_layer: -3,
        output_layer: %{
          size: 1
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
    
    try do
      Neurx.build(%{
        input_layer: "hello",
        output_layer: %{
          size: 1
        }
      })
      assert(false) # should not reach here.
    rescue
      ArgumentError -> nil
    end
  end

  test "Invalid output size." do
    try do
      Neurx.build(%{
        input_layer: 3,
        output_layer: %{
          size: 0
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
    
    try do
      Neurx.build(%{
        input_layer: 2,
        output_layer: %{
          size: "NO"
        }
      })
      assert(false) # should not reach here.
    rescue
      ArgumentError -> nil
    end
  end

  test "Invalid hidden layer size." do
    try do
      Neurx.build(%{
        input_layer: 3,
        output_layer: %{
          size: 1
        },
        hidden_layers: [
          %{
            size: -2
          }
        ]
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
    
    try do
      Neurx.build(%{
        input_layer: 3,
        output_layer: %{
          size: 1
        },
        hidden_layers: [
          %{
            size: "one"
          }
        ]
      })
      assert(false) # should not reach here.
    rescue
      ArgumentError -> nil
    end
  end
  
  test "Unknown Activation function on output layer." do
    try do
      Neurx.build(%{
        input_layer: 3,
        output_layer: %{
          size: 1,
          activation: %{
            type: "SKIDADDLE SKIDOODLE."
          }
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Unknown Activation function on hidden layer." do
    try do
      Neurx.build(%{
        input_layer: 3,
        output_layer: %{
          size: 1
        },
        hidden_layers: [
          %{
            size: 1,
            activation: %{
              type: "SKIDADDLE SKIDOODLE."
            }
          }
        ]
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Unknown loss function." do
    try do
      Neurx.build(%{
        input_layer: 100,
        output_layer: %{
          size: 2
        },
        loss_function: %{
          type: "spaghet"
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Unknown optimization function." do
    try do
      Neurx.build(%{
        input_layer: 100,
        output_layer: %{
          size: 1
        },
        optim_function: %{
          type: "meme review",
          learning_rate: 0.3
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
  end
  
  test "Invalid learning rate." do
    try do
      Neurx.build(%{
        input_layer: 100,
        output_layer: %{
          size: 1
        },
        optim_function: %{
          type: "SGD",
          learning_rate: 0
        }
      })
      assert(false) # should not reach here.
    rescue
      RuntimeError -> nil
    end
    
    try do
      Neurx.build(%{
        input_layer: 100,
        output_layer: %{
          size: 1
        },
        optim_function: %{
          type: "SGD",
          learning_rate: "two"
        }
      })
      assert(false) # should not reach here.
    rescue
      ArithmeticError -> nil
    end
  end
end
