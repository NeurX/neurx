defmodule TestData do

  @simple_training_data [
    %{input: [0,0,1], output: [0]},
    %{input: [1,1,1], output: [1]},
    %{input: [1,0,1], output: [1]},
    %{input: [0,1,1], output: [0]}
  ]

  @simple_test_data [
    %{input: [1,0,0], output: [1]}
  ]
  
  def get_simple_training_data do
    @simple_training_data
  end
  
  def get_simple_test_data do
    @simple_test_data
  end
end
