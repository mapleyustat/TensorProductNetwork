/* File: TensorProd.h
   Author: David Lindberg

   Here I implement functions used to calculate the tensor product of a sentence
   and extract a filler vector from a given tensor product representation
*/

#ifndef TENSORPROD_H
#define TENSORPROD_H

/* function: calcTensorProduct
  input:
        1 pointer to role vector
        2 pointer to filler vector
        3 size of role and filler vectors
  outout:
        1 pointer to 3x3 array representing tensor product of
          role and filler vector
*/
double** calcTensorProduct(double* role, double* filler, int dim)
{
    double** tp = new double*[dim];
    for (int i = 0; i < dim; i++)
    {
        tp[i] = new double[dim];
        memset(tp[i], 0, dim*sizeof(double));
    }

	// here we calculate the tensor product using the 'role' vector,
	// which is defined in main.cpp and passed to this function
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            tp[i][j] = filler[i]*role[j];

    return tp;
    
}

/* function: extractFillerVector
* input:
        1 pointer to tensor product array of size 'dim'x'dim'
        2 pointer to role vector of size 'dim'
        3 'dim'
  output:
        1 pointer to filler vector
 
  NOTE: This function is not currently used. I wrote it to test
		whether I was creating tensor products correctly and could
        successfully extract a sentence from its tensor product
        representation
*/
double* extractFillerVector(double** tp, double* role, int dim)
{
    double* filler = new double[dim];
    memset(filler, 0, sizeof(double));

    //calculate square of role norm
    double norm = 0;
    for (int i = 0; i < dim; i++)
        norm += role[i]*role[i];

    for (int i = 0; i < dim; i++)
    {
        for (int d = 0; d < dim; d++)
            filler[i] += tp[i][d] * role[d];
        filler[i] /= norm;
    }

    // filler now holds the codes representing the subj, verb, and obj of 
    // a sentence

    return filler;
}
#endif
