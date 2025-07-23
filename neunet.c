#include "neunet.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Xavier normal (Gaussian)
double randNormalized() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }

// Init weights
double xavierInit(int inputs) {
  double stddev = sqrt(1.0 / inputs);
  return randNormalized() * stddev;
}

// Initializers
Neuron *initNeuron(int inputs) {
  double weight = xavierInit(inputs);

  Neuron *n = (Neuron *)malloc(sizeof(Neuron));
  n->bias = BIAS;
  n->weights = (double *)malloc(sizeof(double) * inputs);

  for (int i = 0; i < inputs; i++) {
    n->weights[i] = weight;
  }

  return n;
}

Layer *initLayer(int neurons, int neuronsInPreviousLayer) {
  Layer *l = (Layer *)malloc(sizeof(Layer));

  l->neuronCount = neurons;
  l->neurons = (Neuron **)malloc(sizeof(Neuron *) * neurons);

  for (int i = 0; i < neurons; i++)
    l->neurons[i] = initNeuron(neuronsInPreviousLayer);

  return l;
}

Model *initModel(int layers, int *neuronsInLayers) {
  Model *m = (Model *)malloc(sizeof(Model));

  m->layerCount = layers;
  m->layers = (Layer **)malloc(sizeof(Layer *) * layers);

  m->layers[0] = initLayer(neuronsInLayers[0], 0);

  for (int i = 1; i < layers; i++)
    m->layers[i] = initLayer(neuronsInLayers[i], neuronsInLayers[i - 1]);

  return m;
}

// Handling of chosen activation function
int activationFunctionToUse = 0;

const char *possibleActivationFunctions[] = {"Sigmoid",
                                             "Hyperbolic Tangent",
                                             "Rectified Linear Unit (ReLU)",
                                             "Leaky rectified linear unit (Leaky ReLU)",
                                             "Sigmoid Linear Unit (SiLU)",
                                             "Exponential Linear Sigmoid SquasHing (ELiSH)"};

double actFunction(double x) {
  switch (activationFunctionToUse) {
  case sigmoid:
    return 1 / (1 + exp(-x));

  case tangent:
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));

  case ReLU:
    return fmax(0, x);

  case leakyReLU:
    return x > 0 ? x : 0.01 * x;

  case SiLU:
    return x / (1 + exp(-x));

  case ELiSH:
    return x < 0 ? (exp(x) - 1) / (1 + exp(-x)) : x / (1 + exp(-x));

  default:
    printf("Unrecognized function!");
    exit(EXIT_FAILURE);
  }
}

double actFunctionDerivative(double x) {
  switch (activationFunctionToUse) {
  case sigmoid: {
    double s = (1 / (1 + exp(-x)));
    return s * (1 - s);
  }

  case tangent:
    return 1 - pow((exp(x) - exp(-x)) / (exp(x) + exp(-x)), 2);

  case ReLU:
    return x > 0 ? 1 : 0;

  case leakyReLU:
    return x > 0 ? 1 : 0.01;

  case SiLU:
    return (1 + exp(-x) + x * exp(-x)) / pow(1 + exp(-x), 2);

  case ELiSH:
    return x < 0 ? ((2 * exp(2 * x) + exp(3 * x) - exp(-x)) / (exp(2 * x) + 2 * exp(x) + 1))
                 : ((x * exp(x) + exp(2 * x) + exp(x)) / (exp(2 * x) + 2 * exp(x) + 1));

  default:
    printf("Unrecognized function!");
    exit(EXIT_FAILURE);
  }
}

// Forward propagation
void calcNeuronValue(Neuron *n, Layer *l) {
  double output = 0.0;

#pragma omp parallel for reduction(+ : output)
  for (int i = 0; i < l->neuronCount; i++) {
    output += n->weights[i] * l->neurons[i]->activatedValue;
  }

  n->value = output + n->bias;
  n->activatedValue = actFunction(n->value);
}

void applySoftmax(double *z, double *a, int size) {
  double max = z[0];
  for (int i = 1; i < size; ++i)
    if (z[i] > max)
      max = z[i];

  double sum_exp = 0.0;
  for (int i = 0; i < size; ++i) {
    a[i] = exp(z[i] - max);
    sum_exp += a[i];
  }

  for (int i = 0; i < size; ++i)
    a[i] /= sum_exp;
}

double *normalizeOutput(Layer *l) {
  double *input = (double *)malloc(sizeof(double) * l->neuronCount);
  double *output = (double *)malloc(sizeof(double) * l->neuronCount);

  for (int i = 0; i < l->neuronCount; i++)
    input[i] = l->neurons[i]->activatedValue;

  applySoftmax(input, output, l->neuronCount);
  free(input);

  return output;
}

double crossEntropyLoss(double *predicted, int trueLabel, int outputSize) {
  double epsilon = 1e-12;
  return -log(predicted[trueLabel] + epsilon);
}

double *forwardPropagation(Model *m, double *trainValues) {
  for (int i = 0; i < m->layers[0]->neuronCount; i++) {
    m->layers[0]->neurons[i]->value = trainValues[i];
    m->layers[0]->neurons[i]->activatedValue = trainValues[i];
  }

  for (int i = 1; i < m->layerCount - 1; i++) {
#pragma omp parallel for
    for (int j = 0; j < m->layers[i]->neuronCount; j++) {
      calcNeuronValue(m->layers[i]->neurons[j], m->layers[i - 1]);
    }
  }

  Layer *lastLayer = m->layers[m->layerCount - 1];

#pragma omp parallel for
  for (int j = 0; j < lastLayer->neuronCount; j++) {
    calcNeuronValue(lastLayer->neurons[j], m->layers[m->layerCount - 2]);
    lastLayer->neurons[j]->activatedValue = lastLayer->neurons[j]->value;
  }

  double *logits = (double *)malloc(sizeof(double) * lastLayer->neuronCount);
  for (int j = 0; j < lastLayer->neuronCount; j++) {
    logits[j] = lastLayer->neurons[j]->activatedValue;
  }

  double *predictedValues = (double *)malloc(sizeof(double) * lastLayer->neuronCount);
  applySoftmax(logits, predictedValues, lastLayer->neuronCount);

  for (int j = 0; j < lastLayer->neuronCount; j++) {
    lastLayer->neurons[j]->activatedValue = predictedValues[j];
  }

  free(logits);

  return predictedValues;
}

// Backward propagation
void computeDeltaOutputLayer(Layer *outputLayer, double *expected, double *delta) {
  for (int i = 0; i < outputLayer->neuronCount; i++) {
    double predicted = outputLayer->neurons[i]->activatedValue;
    delta[i] = predicted - expected[i];
  }
}

void computeDeltaHiddenLayer(Layer *current, Layer *next, double *deltaCurrent, double *deltaNext) {
  for (int i = 0; i < current->neuronCount; i++) {
    double sum = 0.0;
    for (int j = 0; j < next->neuronCount; j++) {
      sum += deltaNext[j] * next->neurons[j]->weights[i];
    }
    double derivative = actFunctionDerivative(current->neurons[i]->value);
    deltaCurrent[i] = derivative * sum;
  }
}

void updateWeightsAndBias(Layer *prev, Layer *curr, double *delta, double learningRate) {
#pragma omp parallel for
  for (int i = 0; i < curr->neuronCount; i++) {
    for (int j = 0; j < prev->neuronCount; j++) {
      double input = prev->neurons[j]->activatedValue;
      curr->neurons[i]->weights[j] -= learningRate * delta[i] * input;
    }
    curr->neurons[i]->bias -= learningRate * delta[i];
  }
}

void backwardPropagation(Model *m, double *expected, double learningRate) {
  int L = m->layerCount;

  double **deltas = (double **)malloc(sizeof(double *) * L);
  for (int i = 0; i < L; i++) {
    deltas[i] = (double *)calloc(m->layers[i]->neuronCount, sizeof(double));
  }

  computeDeltaOutputLayer(m->layers[L - 1], expected, deltas[L - 1]);

  for (int k = L - 2; k > 0; k--) {
    computeDeltaHiddenLayer(m->layers[k], m->layers[k + 1], deltas[k], deltas[k + 1]);
  }

  for (int k = 1; k < L; k++) {
    updateWeightsAndBias(m->layers[k - 1], m->layers[k], deltas[k], learningRate);
  }

  for (int i = 0; i < L; i++) {
    free(deltas[i]);
  }

  free(deltas);
}

// Main training function
void printProgressBar(int currentSample, int totalSamples) {
  int barWidth = 20;

  double progress = (double)(currentSample + 1) / totalSamples;
  int pos = (int)(barWidth * progress);

  printf("[");
  for (int i = 0; i < barWidth; i++) {
    if (i < pos)
      printf("=");
    else
      printf(" ");
  }
  printf("] %d%%", (int)(progress * 100));
  fflush(stdout);
}

void trainModel(Model *m, double **trainSet, int trainSize, int *trainLabels, int outputs, int epochs, double learningRate) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0.0;

    for (int sample = 0; sample < trainSize; sample++) {
      double expected[outputs];

      for (int k = 0; k < outputs; k++)
        expected[k] = (trainLabels[sample] == k) ? 1.0 : 0.0;

      double *predicted = forwardPropagation(m, trainSet[sample]);

      totalLoss += crossEntropyLoss(predicted, trainLabels[sample], outputs);

      backwardPropagation(m, expected, learningRate);

      if (sample % 100 == 0 || sample == trainSize - 1) {
        printf("\rWorking on epoch %d ", epoch + 1);
        printProgressBar(sample, trainSize);
      }
    }

    printf("\nEpoch %d/%d - Average Loss: %f\n\n", epoch + 1, epochs, totalLoss / trainSize);
  }
}

// Calculate model accuracy
double calculateAccuracy(Model *m, double **testSet, int testSize, int *testLabels, int outputs) {
  printf("\n\nStarting to load values...\n");

  int correctGuesses = 0;

  for (int i = 0; i < testSize; i++) {
    double *predicted = forwardPropagation(m, testSet[i]);

    int guess = 0;

    for (int j = 1; j < outputs; j++) {
      if (predicted[guess] < predicted[j])
        guess = j;
    }

    if (guess == testLabels[i])
      correctGuesses++;

    if (i % 100 == 0 || i == testSize - 1) {
      printf("\rWorking on training ");
      printProgressBar(i, testSize);
    }
  }

  printf("\n\nCorrect guesses: %d", correctGuesses);

  return (double)correctGuesses / testSize;
}

// Test model on image
int recognizeNumber(Model *m, double *img, int outputs) {
  double *predicted = forwardPropagation(m, img);

  int guess = 0;

  for (int j = 1; j < outputs; j++) {
    if (predicted[guess] < predicted[j])
      guess = j;
  }

  return guess;
}

// Free model
void freeModel(Model *model) {
  if (model == NULL)
    return;

  for (int i = 0; i < model->layerCount; i++) {
    Layer *layer = model->layers[i];

    if (layer != NULL) {
      for (int j = 0; j < layer->neuronCount; j++) {
        Neuron *neuron = layer->neurons[j];

        if (neuron != NULL) {
          free(neuron->weights);
          free(neuron);
        }
      }
      free(layer->neurons);
      free(layer);
    }
  }

  free(model->layers);
  free(model);
}

// Save model
void saveModel(Model *m, char *fileName, int activationFunctionToUse) {
  int fileDescriptor = open(fileName, O_CREAT | O_WRONLY, 0777);

  int layerCount = m->layerCount;
  int *neuronsInLayers = (int *)malloc(sizeof(int) * layerCount);
  dprintf(fileDescriptor, "%d - ", layerCount);

  for (int i = 0; i < layerCount; i++) {
    dprintf(fileDescriptor, "%d ", m->layers[i]->neuronCount);
    neuronsInLayers[i] = m->layers[i]->neuronCount;
  }

  dprintf(fileDescriptor, "%d\n\n", activationFunctionToUse);

  for (int i = 1; i < layerCount; i++) {
    for (int j = 0; j < neuronsInLayers[i]; j++) {
      for (int k = 0; k < m->layers[i - 1]->neuronCount; k++) {
        dprintf(fileDescriptor, "%.17g ", m->layers[i]->neurons[j]->weights[k]);
      }

      dprintf(fileDescriptor, "%.17g\n", m->layers[i]->neurons[j]->bias);
    }

    dprintf(fileDescriptor, "\n");
  }
}

// Load model
double *getNeuronValues(FILE *fp, int neurons, double *bias) {
  char *line = NULL;
  size_t len = 0;
  ssize_t read = getline(&line, &len, fp);

  if (read == -1) {
    free(line);
    return NULL;
  }

  double *values = malloc(sizeof(double) * neurons);
  int count = 0;

  char *token = strtok(line, " \t\n");
  while (token != NULL && count < neurons) {
    char *endptr;
    values[count++] = strtod(token, &endptr);
    token = strtok(NULL, " \t\n");
  }

  if (token != NULL) {
    char *endptr;
    *bias = strtod(token, &endptr);
  } else {
    *bias = 0.0;
  }

  free(line);
  return values;
}

void loadModel(Model **m, char *fileName, int *outputs, int *activationFunctionToUse) {
  FILE *fp = fopen(fileName, "r");

  if (!fp) {
    perror("File not found!");
    exit(EXIT_FAILURE);
  }

  char *line = NULL;
  size_t len = 0;
  ssize_t read = getline(&line, &len, fp);

  if (read == -1) {
    printf("Error during read\n");
    free(line);
    fclose(fp);
    exit(EXIT_FAILURE);
  }

  char layerbuffer[16];
  int i = 0;

  while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0' && i < 15) {
    layerbuffer[i] = line[i];
    i++;
  }
  layerbuffer[i] = '\0';

  int layers = atoi(layerbuffer);

  int *neuronsInLayers = (int *)malloc(sizeof(int) * (layers));

  for (int k = 0; k < layers; k++) {
    int bufferSize = 16;
    char buffer[bufferSize];
    int j = 0;

    while (line[i] == ' ' || line[i] == '-')
      i++;

    while (line[i] >= '0' && line[i] <= '9' && j < bufferSize - 1) {
      buffer[j++] = line[i++];
    }
    buffer[j] = '\0';

    neuronsInLayers[k] = atoi(buffer);
  }

  char activationFunctionBuffer[16];
  int j = 0;

  while (line[i] == ' ' || line[i] == '-')
    i++;

  while (line[i] != ' ' && line[i] != '\n' && line[i] != '\0' && j < 15) {
    activationFunctionBuffer[j++] = line[i++];
  }
  activationFunctionBuffer[j] = '\0';

  *activationFunctionToUse = atoi(activationFunctionBuffer);
  printf("\nRecognized this model:");

  printf("\nInputs: %d", neuronsInLayers[0]);
  printf("\nOutputs: %d", neuronsInLayers[layers - 1]);

  printf("\nNeurons in hidden layers: [");
  for (int i = 1; i < layers - 1; i++) {
    printf("%d", neuronsInLayers[i]);

    if (i != layers - 2)
      printf(", ");
  }
  printf("]");

  printf("\nActivation function found in file: %s", activationFunctionBuffer);
  *m = initModel(layers, neuronsInLayers);

  free(line);

  for (int i = 1; i < layers; i++) {
    read = getline(&line, &len, fp);

    for (int j = 0; j < neuronsInLayers[i]; j++) {
      double bias;
      double *weights = getNeuronValues(fp, neuronsInLayers[i - 1], &bias);

      for (int k = 0; k < neuronsInLayers[i - 1]; k++)
        (*m)->layers[i]->neurons[j]->weights[k] = weights[k];

      (*m)->layers[i]->neurons[j]->bias = bias;

      free(weights);
    }
  }

  *outputs = neuronsInLayers[layers - 1];
  fclose(fp);
}
