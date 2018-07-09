defmodule Neurx.NeuronTest do
  use ExUnit.Case
  doctest Neurx.Neuron

  alias Neurx.{Neuron, Connection, Activators, Optimizers}

  test "has default values as an agent" do
    {:ok, pid} = Neuron.start_link()
    neuron = Neuron.get(pid)

    assert(neuron.input == 0)
    assert(neuron.output == 0)
    assert(neuron.incoming == [])
    assert(neuron.outgoing == [])
    assert(neuron.bias? == false)
    assert(neuron.delta == 0)
    assert(neuron.delta_fn == nil)
    assert(neuron.activation_fn == nil)
    assert(neuron.learning_rate == 0.1)
    assert(neuron.optim_fn == nil)
  end

  test "has values passed in as an agent" do
    {:ok, pid} =
      Neuron.start_link(%{
        input: 1,
        output: 2,
        incoming: [1],
        outgoing: [2],
        bias?: true,
        delta: 1,
        delta_fn: 6,
        activation_fn: 66,
        learning_rate: 666,
        optim_fn: 6666
      })

    neuron = Neuron.get(pid)
    assert(neuron.input == 1)
    assert(neuron.output == 2)
    assert(neuron.incoming == [1])
    assert(neuron.outgoing == [2])
    assert(neuron.bias? == true)
    assert(neuron.delta == 1)
    assert(neuron.delta_fn == 6)
    assert(neuron.activation_fn == 66)
    assert(neuron.learning_rate == 666)
    assert(neuron.optim_fn == 6666)
  end


  test "update neuron values" do
    {:ok, pid} = Neuron.start_link()

    Neuron.update(pid, %{
      input: 1,
      output: 2,
      incoming: [1],
      outgoing: [2],
      bias?: true,
      delta: 1,
      delta_fn: 6,
      activation_fn: 66,
      learning_rate: 666,
      optim_fn: 6666
    })

    neuron = Neuron.get(pid)
    assert(neuron.input == 1)
    assert(neuron.output == 2)
    assert(neuron.incoming == [1])
    assert(neuron.outgoing == [2])
    assert(neuron.bias? == true)
    assert(neuron.delta == 1)
    assert(neuron.delta_fn == 6)
    assert(neuron.activation_fn == 66)
    assert(neuron.learning_rate == 666)
    assert(neuron.optim_fn == 6666)
  end

  test "bias neuron" do
    {:ok, pid} = Neuron.start_link(%{bias?: true})
    bias_neuron = Neuron.get(pid)
    assert(bias_neuron.bias?)
    assert(bias_neuron.incoming == [])
    assert(bias_neuron.outgoing == [])
  end

  test ".connect" do
    {:ok, pidA} = Neuron.start_link()
    {:ok, pidB} = Neuron.start_link()

    Neuron.connect(pidA, pidB)

    assert(length(Neuron.get(pidA).outgoing) == 1)
    assert(length(Neuron.get(pidB).incoming) == 1)
  end

  test "activate with specified value" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pid} = Neuron.start_link(%{activation_fn: sigmoid_fn})

    pid |> Neuron.activate(1)
    assert(Neuron.get(pid).output == 1)
  end

  test "activate with no value" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pid} = Neuron.start_link(%{activation_fn: sigmoid_fn})

    pid |> Neuron.activate()
    assert(Neuron.get(pid).output == 0)
  end

  test "activate with incoming connections" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pidX} = Neuron.start_link(%{output: 2, activation_fn: sigmoid_fn})
    {:ok, pidY} = Neuron.start_link(%{output: 5, activation_fn: sigmoid_fn})

    {:ok, connection_one_pid} = Connection.start_link(%{source_pid: pidX})
    {:ok, connection_two_pid} = Connection.start_link(%{source_pid: pidY})

    {:ok, pidA} = Neuron.start_link(%{incoming: [connection_one_pid, connection_two_pid],
      activation_fn: sigmoid_fn})

    pidA |> Neuron.activate()
    assert(Neuron.get(pidA).output == 0.9426758241011313)
  end

  test "activate a bias neuron" do
    {:ok, pid} = Neuron.start_link(%{bias?: true})
    pid |> Neuron.activate()
    assert(Neuron.get(pid).output == 1)
  end

  test "connect and activate two neurons" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    assert(sigmoid_fn)

    {:ok, pidA} = Neuron.start_link(%{activation_fn: sigmoid_fn})
    {:ok, pidB} = Neuron.start_link(%{activation_fn: sigmoid_fn})
    Neuron.connect(pidA, pidB)

    pidA |> Neuron.activate(2)
    pidB |> Neuron.activate()

    neuronA = Neuron.get(pidA)
    neuronB = Neuron.get(pidB)

    assert(neuronA.input == 2)
    assert(neuronA.output == 2)
    assert(neuronB.input == 0.8)
    assert(neuronB.output == 0.6899744811276125)
  end

  test "train: delta should get smaller (learnin yo!)" do
    sigmoid_fn = Activators.getFunction("Sigmoid")
    sigmoid_derv_fn = Activators.getDeltaFunction("Sigmoid")
    sgd = Optimizers.getFunction("SGD")

    assert(sigmoid_fn)
    assert(sigmoid_derv_fn)
    assert(sgd)

    {:ok, pidA} = Neuron.start_link(%{activation_fn: sigmoid_fn, delta_fn: sigmoid_derv_fn, optim_fn: sgd})
    {:ok, pidB} = Neuron.start_link(%{activation_fn: sigmoid_fn, delta_fn: sigmoid_derv_fn, optim_fn: sgd})
    Neuron.connect(pidA, pidB)

    deltas =
      Enum.map(1..100, fn _ ->
        pidA |> Neuron.activate(2)
        pidB |> Neuron.activate()

        pidB |> Neuron.train(1)
        pidA |> Neuron.train()

        Neuron.get(pidB).delta
      end)

    deltas
    |> Stream.with_index()
    |> Enum.each(fn tuple ->
      {delta, index} = tuple

      if Enum.at(deltas, index + 1) do
        assert(delta < Enum.at(deltas, index + 1))
      end
    end)
  end
end
