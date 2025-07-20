#ifndef NEUNET_H
#define NEUNET_H

#define ACTIVATION_FUNCTIONS 6
#define BIAS 0.01

typedef struct {
  double *weights;
  double bias;
  double value;
  double activatedValue;
} Neuron;

typedef struct {
  int neuronCount;
  Neuron **neurons;
} Layer;

typedef struct {
  Layer **layers;
  int layerCount;
} Model;

typedef enum { sigmoid, tangent, ReLU, leakyReLU, SiLU, ELiSH } Functions;
extern const char *possibleActivationFunctions[];
extern int activationFunctionToUse;

double randNormalized();
double xavierInit(int inputs);
Neuron *initNeuron(int inputs);
Layer *initLayer(int neurons, int neuronsInPreviousLayer);
Model *initModel(int layers, int *neuronsInLayers);

void trainModel(Model *m, double **trainSet, int trainSize, int *trainLabels, int outputs, int epochs, double learningRate);
double calculateAccuracy(Model *m, double **testSet, int testSize, int *testLabels, int outputs);
int recognizeNumber(Model *m, double *img, int outputs);
void freeModel(Model *m);

void saveModel(Model *m, char *fileName);
// void loadModelWeights(Model **m, const char *filename);

#endif
