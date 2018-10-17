#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "net.h"


#define ndata 10000
#define ninput 2
#define noutput 1
#define nlayers 4

int main() {
  NN net;
  //double dataset[ndata][ninput];
  //double target[ndata][noutput];
  double **dataset, **target;
  double learning_rate, momentum, weight_decay;
  double out[1];
  int Niterations, batchsize;
  int i, j, count;
  double x, y, t;
  FILE *f;

  // init seed for random generator
  srand48(99);

  // init neural network
  init_net(&net, "tanh", "linear", nlayers, ninput, 8, 8, noutput);
  print_network_structure(&net);
  save_net(&net, "net1.txt");

  // build dataset
  dataset = (double **) malloc(ndata * sizeof(double*));
  if(dataset == NULL) {
    printf("\nERROR: Malloc of dataset failed.\n");
    exit(1);
  }
  for(i=0; i<ndata; i++) {
    dataset[i] = (double *) malloc(ninput * sizeof(double));
    if(dataset[i] == NULL) {
      printf("\nERROR: Malloc of dataset failed.\n");
      exit(1);
    }
  }

  target = (double **) malloc(ndata * sizeof(double*));
  if(target == NULL) {
    printf("\nERROR: Malloc of dataset failed.\n");
    exit(1);
  }
  for(i=0; i<ndata; i++) {
    target[i] = (double *) malloc(noutput * sizeof(double));
    if(target[i] == NULL) {
      printf("\nERROR: Malloc of dataset failed.\n");
      exit(1);
    }
  }

  count = 0;
  for(i=0; i<100; i++) {
    x = -5.0+i*0.1;
    for(j=0; j<100; j++) {
      y = -5.0+j*0.1;
      t = exp(-0.5*(x*x+y*y))/2.0*M_PI;
      dataset[count][0] = x;
      dataset[count][1] = y;
      target[count][0] = t;
      count++;
    }
  }

  // init training
  init_training(&net);
  learning_rate = 0.01;
  momentum = 0.9;
  weight_decay = 0.00001;
  batchsize = 10;
  Niterations = 100;

  train(&net, ndata, dataset, target, learning_rate, momentum, weight_decay, batchsize, Niterations);


  if((f = fopen("output.dat","w")) == NULL) {
    printf("\nERROR while opening output file\n");
    exit(0);
  }
  for(i=0; i<ndata; i++) {
    predict(&net, dataset[i], out);
    fprintf(f, "%lg  %lg %lg %lg\n", dataset[i][0], dataset[i][1], out[0], target[i][0]);
  }
  fclose(f);

  save_net(&net, "net2.txt");
  free_net(&net);
  for(i=0; i<ndata; i++) {
    free(dataset[i]);
    free(target[i]);
  }
  free(dataset);
  free(target);


  return 0;
}
