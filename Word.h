/* File: Word.h
   Author: David Lindberg

   Here I define a 'Word' structure and a function to print a Word's
   string and code value
*/

#ifndef WORD_H
#define WORD_H

#include <string>

typedef struct Word
{
    std::string word;
    double value; // referred to as the word's 'code' in documentation
} Word;


/*function: printWord
*
* input:
*      1 Word to be printed
* output:
*       prints the Word's string and value to the console
*/
void printWord(Word w)
{
    std::cout << w.word << ' ' << w.value << std::endl;
}

#endif
