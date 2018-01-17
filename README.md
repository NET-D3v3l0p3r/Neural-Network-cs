# Neural-Network-cs
This repository is a direct port from [Shiffman's Simple JS Neural Network](https://github.com/shiffman/Neural-Network-p5). Rewritten in c# and compiled to dll.

# Use
```cs

// Creating a Neural Network with # of inputs, hidden neurons, and outputs
int inputs = 4;
int hidden = 16;
int outputs = 2;

var nn = new NeuralNetwork(inputs, hidden, outputs);

// Training the Neural Network with inputs and known outputs
var inputs = new List<double>
{
	-0.3,
 	0.5,
 	0.3,
 	0.2
}

var targets = new List<double> 
{
	0.99, 
	0.01
}

nn.Train(inputs, targets);

// Querying the Neural Network with inputs
var inputs = new List<double>
{
	-0.3,
 	0.5,
  	0.3,
 	0.2
}

var prediction = nn.Query(inputs);

```

By default, the library will use a sigmoid activation function. However, you can select other activation functions as follows (tanh only at the moment)):

```cs
var nn = new NeuralNetwork(inputs, hidden, outputs, "sigmoid");
var nn = new NeuralNetwork(inputs, hidden, outputs, "tanh");
```

> License extends that of the original repository by Daniel Shiffman. 