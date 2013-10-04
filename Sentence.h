/* File: Sentence.h
   Author: David Lindberg

   This is a simple structure used to represent sentences. I also include here
   a function that prints information about a sentence.
*/

#ifndef SENTENCE_H
#define SENTENCE_H

#include <string>
#include <iostream>
#include <iomanip>

using namespace std;

typedef struct Sentence
{
    std::string subj; // subject
    std::string verb; // verb (clearly)
    std::string obj; // object
    double inputTP[3][3]; // tensor product representation of the sentence;
    double targetTP[3][3]; // tensor product we want the backprop network to produce

} Sentence;

/* function: printSentence
   input: a single Sentence object
   output: prints the following
           1. the sentence string
           2. the tensor product representation of the sentence
           3. the target tensor product for the sentence
*/
void printSentence(Sentence s)
{
    cout << s.subj << ' ' << s.verb << ' ' << s.obj << endl << endl;
    cout << "Input TP: " << setw(23) << "Target TP" << endl;
   
    cout.width(3);
    cout << setprecision(4);
    for (int i = 0; i < 3; i++)
    {
        
        cout << fixed <<  s.inputTP[i][0] << ' ' <<  s.inputTP[i][1] << ' ' << s.inputTP[i][2] << '\t'
             << s.targetTP[i][0] << ' ' << s.targetTP[i][1] << ' ' << s.targetTP[i][2] << endl;
    }
    cout << std::endl;
}
#endif
