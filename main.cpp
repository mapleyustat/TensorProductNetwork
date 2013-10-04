/* File: main.cpp
   Author: David Lindberg

   This file contains the main() function that uses my backpropagation network
   to conduct two experiments.

   I define several other functions that assist in the preparation and execution
   of the two experiments and report on the results.
 
 
 */

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "Word.h"
#include "Sentence.h"
#include "TensorProd.h"
#include "Network.h"
using namespace std;

/* BEGIN GLOBAL DECLARATIONS */

double role[3] = {0.75, 0.5, 0.3}; // role vector will be the same for both experiments

/* END CONST DECLARATIONS */


/* BEGIN FUNCTION PROTOTYPES */

void loadData(vector<Word>&, vector<Word>&, vector<Sentence>&, vector<Sentence>&);
void prepExpData(vector<Word>&, vector<Sentence>&, vector<Sentence>&);
double getValue(string, vector<Word>&);
void doExperiment(Network*, vector<Sentence>&, vector<Sentence> &, string, string);
void printResults(Sentence, double**, double&);

/* END FUNCTION PROTOTYPES */


/* MAIN */
int main(int argc, char** argv)
{
    Network* network = new Network();
    vector<Word> exp1; // experiment 1 words and their corresponding values
    vector<Word> exp2; // experiment 2...
    vector<Sentence> training; // training sentences
    vector<Sentence> test; // test sentences;
    
    loadData(exp1, exp2, training, test);
    
    cout << "preparing data for experiment 1... " << endl;
    prepExpData(exp1, training, test);

    network->setLearningRate(LEARN_RATE);
    
    cout << "starting experiment 1..." << endl;
    doExperiment(network, training, test, "hidden_layer_activation", "output_tensor_product");
   
    delete network;
    network = new Network();
    network->setLearningRate(LEARN_RATE);
    cout << "preparing data for experiment 2..." << endl;
    prepExpData(exp2, training, test);
    
    cout << "starting experiment 2..." << endl;
    doExperiment(network, training, test, "hidden_layer_activation2", "output_tensor_product");
    delete network; 
    return 0;
}


/* BEGIN FUNCTION DEFINITIONS */

/* function: loadData
*  behaviour: loads testing and training data for the two experiments
*  input: vectors to be filled with the following data
*  
*	    1.  experiment 1 word values
*       2.  experiment 2 word values
*       3.  training sentences
*       4.  test sentences  
*
*  output: none
*  postcondition: vectors contain the required data
*/
void loadData(vector<Word> &exp1Values, vector<Word> &exp2Values, vector<Sentence> &train, vector<Sentence> &test)
{
    ifstream wordFile;
    Word newWord;
    Sentence newSentence;
    
    // load words and values for exp1
    wordFile.open("first");

    cout << "loading experiment 1 data..." << endl;
    while (!wordFile.eof())
    {
        wordFile >> newWord.word >> newWord.value;
        exp1Values.push_back(newWord);
    }
    wordFile.close();
	
    // load words and values for exp2
    wordFile.open("second");

    cout << "loading experiment 2 data..." << endl;
    while (!wordFile.eof())
    {
        wordFile >> newWord.word >> newWord.value;
        exp2Values.push_back(newWord);
    }
    wordFile.close();
	
    // load training sentences
    wordFile.open("train");

    cout << "loading training sentences..." << endl;
    
    while (!wordFile.eof())
    {
        wordFile >> newSentence.subj >> newSentence.verb >> newSentence.obj;
        train.push_back(newSentence);
    }
    wordFile.close();

    // load test sentences
    wordFile.open("test");

    cout << "loading test sentences..." << endl;
    while (!wordFile.eof())
    {
        wordFile >> newSentence.subj >> newSentence.verb >> newSentence.obj;
        test.push_back(newSentence);
    }
    wordFile.close();
    cout << "data successfully loaded" << endl;
}

/* function: prepExpData
/* input:
*        1. experiment-specfic words
*        2. training sentences
*        3. test sentences
*  output: none
*  postcondition: input and target tensor products for training and test sentences have been calculated
*/
void prepExpData(vector<Word>& words, vector<Sentence>& train, vector<Sentence>& test)
{
    double** tp = NULL;
    double** target = NULL;
    double filler[3];
    double swap;

    for (int i = 0; i < train.size(); i++)
    {
        if (target != NULL)
        {
            for (int j = 0; j < 3; j++)
            {
                delete [] tp[j];
                delete [] target[j];
            }
            delete [] tp;
            tp = NULL;
            delete [] target;
            target = NULL;
        }
        memset(filler, 0, 3*sizeof(double));
		
		// determine the filler vector representation of the sentence
        filler[0] = getValue(train[i].subj, words);
        filler[1] = getValue(train[i].verb, words);
        filler[2] = getValue(train[i].obj, words);
        
        
        tp = calcTensorProduct(role, filler, 3);
        
        swap = filler[2];
        filler[2] = filler[0];
        filler[0] = swap;

        target = calcTensorProduct(role, filler, 3);

        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                train[i].inputTP[r][c] = tp[r][c];
                train[i].targetTP[r][c] = target[r][c];
            }
    }

    for (int i = 0; i < test.size(); i++)
    {
        if (target != NULL)
        {
            for (int j = 0; j < 3; j++)
            {
                delete []  tp[j];
                delete [] target[j];
            }
            delete [] tp;
            tp = NULL;
            delete [] target;
            target = NULL;
        }
   
        memset(filler, 0, 3*sizeof(double));
        filler[0] = getValue(test[i].subj, words);
        filler[1] = getValue(test[i].verb, words);
        filler[2] = getValue(test[i].obj, words);

        tp = calcTensorProduct(role, filler, 3);
        
        swap = filler[2];
        filler[2] = filler[0];
        filler[0] = swap;

        target = calcTensorProduct(role, filler, 3); 
                
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                test[i].inputTP[r][c] = tp[r][c];
                test[i].targetTP[r][c] = target[r][c];
            }
    }    
}

/* function: getValue
*  input:
*        1. string whose value we need
*        2. Word vector
*  output:
*        1. value associated with the input string or -1 if word not found
*/
double getValue(string s, vector<Word>& words)
{
	
	// I use a brute-force linear search here because the set of words
	// is known to be small. If I were to scale this up to a very large
	// vocabulary, I would replace this with a hash table
	
    for (int i = 0; i < words.size(); i++)
        if (s == words[i].word)
            return words[i].value;

    // word not found
    return -1.0;
}

/* function: doExperiment
   input: 
          1 pointer to the network on which the experiment will be run
          2 vector of training Sentences
          3 vector of test Sentences
          4 filename under which to store hidden node activations from the test phase
   output: 
          printSentence and printResults are called to print test results for each sentence
   behaviour:
          initiates and coordinates the training and testing of the network
*/ 
void doExperiment(Network* network, vector<Sentence> &train, vector<Sentence> &test, string hiddenActFileName, string outputFileName)
{
    double*** input = new double**[train.size()]; // array of input tensor products
    double*** target = new double**[train.size()]; // array of output tensor products

    for (int i = 0; i < train.size(); i++)
    {
        input[i] = new double*[3];
        target[i] = new double*[3];
        for (int j = 0; j < 3; j++)
        {
            input[i][j] = new double[3];
            target[i][j] = new double[3];
        }
    }

	// startTraining requires an array of tensor products, so I copy from each training sentence
    for (int i = 0; i < train.size(); i++)
    {
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
            {
                input[i][j][k] = train[i].inputTP[j][k];
                target[i][j][k] = train[i].targetTP[j][k];
            }
        
    }

    cout << "Training network..." << endl;
    network->startTraining(input, target, train.size());

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            delete [] input[i][j];
            delete [] target[i][j];
        }
        delete [] input[i];
        delete [] target[i];
    }
    delete [] input;
    delete [] target;;

    cout << "Training complete" << endl;
  
    input = new double**[test.size()];
   
    for (int i = 0; i < test.size(); i++)
    {
        input[i] = new double*[3];
        for (int j = 0; j < 3; j++)
            input[i][j] = new double[3];

        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                input[i][j][k] = test[i].inputTP[j][k];
    }

    cout << "Entering testing phase..." << endl;
    double*** outputs = network->startTesting(input, test.size(),  hiddenActFileName, outputFileName);

    cout << "Testing complete" << endl << endl << "********RESULTS********" << endl << endl;

    double error = 0; // used to calculate average average error
    for (int i = 0; i < test.size(); i++)
    {
        printSentence(test[i]);
        
        printResults(test[i], outputs[i], error);
        cout << "--------------------" << endl;
    }
    
    cout << "on average, average relative error for each sentence is: " << error/test.size()*100 << " %" 
         << endl << endl;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            delete [] input[i][j];
            
        delete [] input[i];
    }
    delete [] input;

    for (int i = 0; i < test.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            delete [] outputs[i][j];
        }
        delete [] outputs[i];
    }
    delete [] outputs;
}

/* function: printResults
   input:
         1 a single Sentence
         2 network output for the sentence
         3 a variable in which to accumulate the total error 
   output: prints
         1 network output
		 2 relative error for each output node
         3 average relative error over all output nodes for the given sentence
   postcondition:
		 totalErrorAvg is increased by the total error for the given sentence
   NOTE: totalErrorAvg is a misnomer here. This should be renamed 'totalError', as
		 the averaging of this error is done outside this function
 */
void printResults(Sentence s, double** output, double & totalErrorAvg)
{
    double avgRelErr = 0;
    double one, two, three;

    cout << setprecision(4);
    cout.width(5);
    cout << "Network Output: " << setw(22) << "Relative Error" << endl << endl;
    for (int i = 0; i < 3; i++)
    { 
        
        cout << output[i][0] << ' ' << output[i][1] << ' ' << output[i][2] << '\t';
        one = fabs( (s.targetTP[i][0] - output[i][0])/s.targetTP[i][0] );
        avgRelErr += one;
        cout << one*100 << " % ";
        two = fabs( (s.targetTP[i][1] - output[i][1])/s.targetTP[i][1] );
        avgRelErr += two; 
        cout << two*100 << " % ";
        three = fabs( (s.targetTP[i][2] - output[i][2])/s.targetTP[i][2] );
        avgRelErr += three;
        cout << three*100  << " %" << endl;
    }
    totalErrorAvg += avgRelErr/9;
    cout << endl << "Average Relative Error: " << avgRelErr/9*100 << " %" << endl << endl;
}
/* END FUNCTION DEFINITIONS */
