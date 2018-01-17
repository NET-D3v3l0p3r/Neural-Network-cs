// Port: Guy Ross
// Original: Daniel Shiffman
//
// 1/ 17/ 18
//
// Neural Network Direct Port from Daniel Shiffmans Neural Network Library.
// Ported into c# from p5 js.

using System;
using System.Collections.Generic;

namespace CSharpNeuralNetwork
{
    public class NeuralNetwork
    {
        public Int32 InNodes { get; private set; }
        public Int32 HNodes { get; private set; }
        public Int32 ONodes { get; private set; }
        public Double LearningRate { get; private set; }
        public Matrix Wih { get; private set; }
        public Matrix Who { get; private set; }

        public Func<Double, Double> Activation { get; private set; }
        public Func<Double, Double> Derivative { get; private set; }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input_nodes"></param>
        /// <param name="hide_nodes"></param>
        /// <param name="output_nodes"></param>
        /// <param name="activation"></param>
        /// <param name="learning_rate"></param>
        public NeuralNetwork(Int32 input_nodes, Int32 hide_nodes, Int32 output_nodes, String activation = "tanh", Double learning_rate = 0.1)
        {
            // Number of nodes in layer (input, hidden, output)
            // This network is limited to 3 layers

            InNodes = input_nodes;
            HNodes = hide_nodes;
            ONodes = output_nodes;

            LearningRate = learning_rate;

            // These are the weight matrices
            // wih: weights from input to hidden
            // who: weights from hidden to output
            // weights inside the arrays are w_i_j
            // where link is from node i to node j in the next layer
            // Matrix is rows X columns
            Wih = new Matrix(HNodes, InNodes);
            Who = new Matrix(ONodes, HNodes);

            // Start with random values
            Wih.Randomize();
            Who.Randomize();

            // Activation Function
            if (activation == "tanh")
            {
                Activation = Tahn;
                Derivative = DTahnn;
            }
            else
            {
                Activation = Sigmoid;
                Derivative = dSigmoid;
            }

        }

        /// <summary>
        /// Trains the network with inputs and targets.
        /// </summary>
        /// <param name="input_lst"></param>
        /// <param name="targets_lst"></param>
        public void Train(List<Double> input_lst, List<Double> targets_lst)
        {
            // Turn input and target arrays into matrices
            Matrix inputs = Matrix.FromList(input_lst);
            Matrix targets = Matrix.FromList(targets_lst);

            // The input to the hidden layer is the weights (wih) multiplied by inputs
            var hidden_inputs = Matrix.MultiplyMatrices(Wih, inputs);

            // The outputs of the hidden layer pass through sigmoid activation function
            var hidden_outputs = Matrix.Map(hidden_inputs, Activation);

            // The input to the output layer is the weights (who) multiplied by hidden layer
            var output_inputs = Matrix.MultiplyMatrices(Who, hidden_outputs);

            // The output of the network passes through sigmoid activation function
            var outputs = Matrix.Map(output_inputs, Activation);

            // Error is TARGET - OUTPUT
            var output_errors = Matrix.SubtractMatrices(targets, outputs);

            // ~ Back prop.

            // Transpose hidden <-> output weights
            var whoT = Who.Transpose();

            // Hidden errors is output error multiplied by weights (who)
            var hidden_errors = Matrix.MultiplyMatrices(whoT, output_errors);

            // Calculate the gradient, this is much nicer in python!
            var gradient_output = Matrix.Map(outputs, Derivative);
            
            // Weight by errors and learing rate
            gradient_output.Multiply(output_errors);
            gradient_output.Multiply(LearningRate);

            // Gradients for next layer, more back propogation!
            var gradient_hidden = Matrix.Map(hidden_outputs, Derivative);

            // Weight by errors and learning rate
            gradient_hidden.Multiply(hidden_errors);
            gradient_hidden.Multiply(LearningRate);

            // Change in weights from HIDDEN --> OUTPUT
            var trans_hidden_outputs = hidden_outputs.Transpose();
            var deltaw_output = Matrix.MultiplyMatrices(gradient_output, trans_hidden_outputs);
            Who.Add(deltaw_output);

            // Change in weights from INPUT --> HIDDEN
            var trans_inputs = inputs.Transpose();
            var deltaw_hidden = Matrix.MultiplyMatrices(gradient_hidden, trans_inputs);
            Wih.Add(deltaw_hidden);
        }

        /// <summary>
        /// Queries the network.
        /// </summary>
        /// <param name="inputs_lst"></param>
        /// <returns></returns>
        public List<Double> Query(List<Double> inputs_lst)
        {
            var inputs = Matrix.FromList(inputs_lst);

            var hidden_inputs = Matrix.MultiplyMatrices(Wih, inputs);

            var hidden_outputs = Matrix.Map(hidden_inputs, Activation);

            var output_inputs = Matrix.MultiplyMatrices(Who, hidden_outputs);

            var outputs = Matrix.Map(output_inputs, Activation);

            return outputs.ToList();
        }

        #region Static Methods
        public static Double Sigmoid(Double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }

        public static Double dSigmoid(Double value)
        {
            return value * (1 - value);
        }

        public static Double Tahn(Double value)
        {
            return Math.Tanh(value);
        }

        public static Double DTahnn(Double value)
        {
            return 1 / (Math.Pow(Math.Cosh(value), 2));
        }

        public static Double Mutate(Double value)
        {
            if (Convert.ToDouble(new Random().Next(1)) < 0.1)
            {
                var offset = new Random().NextDouble() * (0.1 - -0.1) + -0.1;

                var newx = value + offset;
                return newx;
            }
            else
            {
                return value;
            }
        }
        #endregion
    }
}
