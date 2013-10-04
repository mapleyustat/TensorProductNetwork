/* File: Network.h
   Author: David Lindberg

   This defines a backpropagation network. The constants near the top of the
   file control the number of hidden nodes, training iterations, and learning
   rate. If these are changed, the program must be recompiled.
 
   Much of this is designed ad-hoc for a network with 3x3 input and output.
   Changing the size of the input/output arrays requires modification and
   recompilation.
*/

#ifndef NETWORK_H
#define NETWORK_H

#include <string>

using namespace std;

const int NUM_HIDDEN = 6; // no. of nodes in hidden layer
const int RUNS = 50000; // no. of iterations through training set
const double LEARN_RATE = 0.25; // network learning rate

class Network
{
    public:
        /* method: Constructor
           input: none
           output: none
           postcondition: connection weights are set to random values in range [-0.1,0.1]
        */
        Network(void);

        /* method: giveInput
           input: input tensor product
           output: none
           postcondition: 'hidden' and 'output' nodes reflect the given input;
        */
        void giveInput(double**);

        /* method: startTraining
           input:  
                 1 array of input tensor products
                 2 array of target tensor products
                 3 number of training sentences
           output: none
        */
        void startTraining(double***, double***, int);
    
        /* method: startTesting
           input: 
                 1 array of input tensor products
                 2 name of file in which to store hidden layer activations
                 3 name of file to store output layer activations
           output: 
                 1 array of output tensor products
           postcondition: hidden and output unit acivations have been written to separate files (1 file for all hidden, 1 for
                          all output)
        */
        double*** startTesting(double***, int,  string, string);

        /* method: setLearningRate
           input: 
                 1 real number in range [0,1];
           output: none
           postcondition: 'learningRate' set to the given input value
        */
        void setLearningRate(double);
      
    private:
        /* method: backprop
           input: 
                 1 input tensor product
                 2 difference between output tensor product and target tensor product
           ouput: none
           postcondition: connection weights have been modified
        */
        void backprop(double**, double**);
 
        double hidden[NUM_HIDDEN];
        double inputToHiddenWts[3][3][NUM_HIDDEN];
        double hiddenToOutputWts[NUM_HIDDEN][3][3];
        double output[3][3];;
        double learningRate;
};
#endif
