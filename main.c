#include "mnist.h"
#include "neunet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int getRandomInt(int min, int max) { return min + rand() / (RAND_MAX / (max - min + 1) + 1); }

int main() {
  int layers, inputs, outputs, epochs;

  printf("Number of hidden layers: ");
  scanf("%d", &layers);

  if (layers <= 0) {
    printf("Value not valid!");
    return EXIT_FAILURE;
  } else
    printf("\n");

  int *neuronsInLayers = (int *)malloc(sizeof(int) * (layers + 2));

  printf("Number of inputs: ");
  scanf(" %d", &inputs);
  neuronsInLayers[0] = inputs;

  printf("Number of outputs: ");
  scanf(" %d", &outputs);
  neuronsInLayers[layers + 1] = outputs;

  for (int i = 1; i <= layers; i++) {
    int neurons;

    printf("Number of neurons for hidden layer %d: ", i);
    scanf("%d", &neurons);

    if (neurons <= 0) {
      printf("Value not valid!");
      return EXIT_FAILURE;
    } else {
      neuronsInLayers[i] = neurons;
    }
  }

  printf("Number of epochs: ");
  scanf(" %d", &epochs);

  printf("\nChoose an activation function from the following\n");

  for (int i = 0; i < ACTIVATION_FUNCTIONS; i++) {
    printf("%d - %s\n", i, possibleActivationFunctions[i]);
  }

  printf("Your choice: ");
  scanf(" %d", &activationFunctionToUse);

  if (activationFunctionToUse < 0 || activationFunctionToUse > ACTIVATION_FUNCTIONS) {
    printf("Value not valid!");
    return EXIT_FAILURE;
  } else
    printf("\n");

  Model *myModel = initModel(layers + 2, neuronsInLayers);

  printf("Loading dataset...\n");
  load_mnist();
  printf("Dataset loaded!\n\n");

  double **trainInputs = malloc(NUM_TRAIN * sizeof(double *));
  for (int i = 0; i < NUM_TRAIN; i++) {
    trainInputs[i] = trainImages[i];
  }

  srand(time(NULL) ^ getpid());

  trainModel(myModel, trainInputs, NUM_TRAIN, trainLabels, outputs, epochs, 0.01);

  printf("Train successful! Starting test...\n");

  double **testOutputs = malloc(NUM_TEST * sizeof(double *));
  for (int i = 0; i < NUM_TEST; i++) {
    testOutputs[i] = testImages[i];
  }

  double accuracy = calculateAccuracy(myModel, testOutputs, NUM_TEST, testLabels, outputs);
  printf("\nCalculated accuracy: %.2f", accuracy * 100);
  printf("%%");

  while (1) {
    int imageIndex;
    printf("\n\nInsert a value between 0 and %d, type [-1] to exit: ", NUM_TEST - 1);
    scanf(" %d", &imageIndex);

    if (imageIndex == -1) {
      return EXIT_SUCCESS;
    } else if (imageIndex < 0 || imageIndex >= NUM_TEST) {
      printf("\nValue not valid!");
      return EXIT_FAILURE;
    }

    printf("\n");
    printImage(testImages[imageIndex]);

    int guess = recognizeNumber(myModel, testImages[imageIndex], outputs);
    printf("Label: %d - The model recognized a %d", testLabels[imageIndex], guess);
  }
}
