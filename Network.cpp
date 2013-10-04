/* File: Network.cpp
   Author: David Lindberg

   This is the implementation of the backpropagation network defined in
   Network.h. See that header file for a full description of each method.
*/

#include "Network.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

Network::Network(void)
{
    // weights should be random initially
    srand(time(NULL));

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < NUM_HIDDEN; k++)
            {
				// give each weight a random number between -0.1 and 0.1
                inputToHiddenWts[i][j][k] = (rand() % 200 - 100)/1000.0f;
                hiddenToOutputWts[k][i][j] = (rand() % 200 - 100)/1000.0f;

            }         
}

void Network::giveInput(double** input)
{
    for (int i = 0; i < NUM_HIDDEN; i++)  
        hidden[i] = 0.0;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            output[i][j] = 0;
    
    
    // propagate input activation to hidden layer

    for (int i = 0; i < NUM_HIDDEN; i++)
        for (int j = 0; j < 3; j++)   
            for (int k = 0; k < 3; k++)
                hidden[i] += inputToHiddenWts[j][k][i]*input[j][k];

    // propagate activation from hidden layer to output layer

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < NUM_HIDDEN; k++)
                output[i][j] += hiddenToOutputWts[k][i][j]*hidden[k];
}

void Network::startTraining(double*** inputTP, double*** targetTP, int size)
{
    double** diff = new double*[3];
    for (int i = 0; i < 3; i++)
        diff[i] = new double[3];

    for (int rns = 0; rns < RUNS ; rns++)
    {
        // for each training sentence...
        for (int i = 0; i < size; i++)
        {
            giveInput(inputTP[i]);

			//output layer now relates to the input we just gave
			
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    diff[j][k] = targetTP[i][j][k] - output[j][k];
    
            backprop(inputTP[i], diff);    
        }
    }

    for (int i = 0; i < 3; i++)
        delete [] diff[i];
    delete [] diff;    
}

double*** Network::startTesting(double*** inputTP, int size,  string hiddenActFileName, string outputFileName)
{
    ofstream hiddenActFile;
    ofstream outputFile;
    ofstream weightFile;
 
    hiddenActFile.open(hiddenActFileName.c_str());
    outputFile.open(outputFileName.c_str());

    double*** outputs = new double**[size];
    for (int i = 0; i < size; i++)
    {
        outputs[i] = new double*[3];
        for (int j = 0; j < 3; j++)
            outputs[i][j] = new double[3];
    }
    
    for (int i = 0; i < size; i++)
    {
        giveInput(inputTP[i]);

        // hidden and output layer nodes are changed according to 
		// the input we just gave to the network

        for (int j = 0; j < NUM_HIDDEN; j++)
        {
            hiddenActFile << hidden[j] << " ";
        }

		// I save output activation to a file, but as of now, that file is just for
		// interest
		
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
            {
                outputs[i][j][k] = output[j][k];
                outputFile << output[j][k] << " ";
            }
        
        hiddenActFile << endl;
        outputFile << endl;
    }

    hiddenActFile.close();
    outputFile.close();

	// it might be interesting to see what final weights were learned
	// (this probably should have been printed earlier)
	
    cout << "final learned weights: " << endl;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < NUM_HIDDEN; k++)
                cout << "input (" << i << ", " << j << ") to hidden (" << k << ") = " << inputToHiddenWts[i][j][k] << endl;

    cout << endl;

    for (int k = 0; k < NUM_HIDDEN; k++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cout << "hidden (" << k << ") to output (" << i << ", " << j << ") = " << hiddenToOutputWts[k][i][j] << endl;
    return outputs;
}

void Network::setLearningRate(double rate)
{
    learningRate = rate;
}

void Network::backprop(double** input, double** B)
{   
    double B_h[NUM_HIDDEN];
 
    for (int i = 0; i < NUM_HIDDEN; i++)
        B_h[i] = 0;

    // first: adjust connection weights between hidden layer and output layer
    for (int i = 0; i < NUM_HIDDEN; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
            {
                hiddenToOutputWts[i][j][k] += learningRate*hidden[i]*output[j][k]*(1-output[j][k])*B[j][k];
                if (hiddenToOutputWts[i][j][k] > 1)
                    hiddenToOutputWts[i][j][k] = 1;
                else if (hiddenToOutputWts[i][j][k] < 0)
                    hiddenToOutputWts[i][j][k] = 0;
            }

    // next: calculate benefit of changing each hidden layer unit

    for (int i = 0; i < NUM_HIDDEN; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                B_h[i] += hiddenToOutputWts[i][j][k]*output[j][k]*(1-output[j][k])*B[j][k];

    // last: adjust weights between input and first hidden layer
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < NUM_HIDDEN; k++)
            {
                inputToHiddenWts[i][j][k] += learningRate*input[i][j]*hidden[k]*(1-hidden[k])*B_h[k];
                if (inputToHiddenWts[i][j][k] > 1)
                    inputToHiddenWts[i][j][k] = 1;
                else if (inputToHiddenWts[i][j][k] < 0)
                    inputToHiddenWts[i][j][k] = 0;
            }
}

