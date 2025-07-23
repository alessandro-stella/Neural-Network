#include "mnist.h"
#include "neunet.h"
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAX_NAME_LENGTH 20

double *readPng(char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return NULL;

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);
  if (!png || !info) {
    fclose(fp);
    return NULL;
  }

  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return NULL;
  }

  png_init_io(png, fp);
  png_read_info(png, info);

  png_uint_32 width = png_get_image_width(png, info);
  png_uint_32 height = png_get_image_height(png, info);
  png_byte color_type = png_get_color_type(png, info);
  png_byte bit_depth = png_get_bit_depth(png, info);

  if (width != SIDE_SIZE || height != SIDE_SIZE) {
    fprintf(stderr, "Errore: invalid image size! Required a %dx%d image\n", SIDE_SIZE, SIDE_SIZE);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return NULL;
  }

  if (bit_depth == 16)
    png_set_strip_16(png);

  if (color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if (png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_rgb_to_gray(png, 1, -1, -1);

  if (color_type & PNG_COLOR_MASK_ALPHA)
    png_set_strip_alpha(png);

  png_read_update_info(png, info);

  png_size_t rowbytes = png_get_rowbytes(png, info);
  png_bytep *rows = malloc(sizeof(png_bytep) * height);
  for (png_uint_32 y = 0; y < height; y++)
    rows[y] = malloc(rowbytes);

  png_read_image(png, rows);

  double *grayscale = malloc(sizeof(double) * height * width);
  int index = 0;

  for (int y = 0; y < height; y++) {
    png_bytep row = rows[y];
    for (int x = 0; x < width; x++) {
      png_byte pixel = row[x];
      grayscale[index++] = 1.0 - ((double)pixel / 255.0);
    }
  }

  for (int y = 0; y < height; y++)
    free(rows[y]);
  free(rows);

  png_destroy_read_struct(&png, &info, NULL);
  fclose(fp);

  return grayscale;
}

void recognizeCustomImage(Model *m, int outputs) {
  char fileName[MAX_NAME_LENGTH + 4];

  printf("\nInsert file name: ");
  scanf(" %s", fileName);

  double *img = readPng(fileName);
  printImage(img);

  int guess = recognizeNumber(m, img, outputs);
  printf("The model recognized a %d", guess);
}

void recognizeTestImage(Model *m, int outputs) {
  int imageIndex;
  printf("\n\nInsert a value between 0 and %d, type [-1] to exit: ", NUM_TEST - 1);
  scanf(" %d", &imageIndex);

  if (imageIndex == -1) {
    exit(EXIT_SUCCESS);
  } else if (imageIndex < 0 || imageIndex >= NUM_TEST) {
    printf("\nValue not valid!");
    exit(EXIT_FAILURE);
  }

  printImage(testImages[imageIndex]);

  int guess = recognizeNumber(m, testImages[imageIndex], outputs);
  printf("Label: %d - The model recognized a %d", testLabels[imageIndex], guess);
}

int main(int argc, char *argv[]) {
  Model *myModel, *newModel;
  int layers, inputs, outputs, epochs;

  printf("Loading dataset...\n");
  load_mnist();
  printf("Dataset loaded!\n\n");

  if (argc > 1) {
    loadModel(&myModel, argv[1], &outputs, &activationFunctionToUse);
  } else {
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

    myModel = initModel(layers + 2, neuronsInLayers);

    double **trainInputs = malloc(NUM_TRAIN * sizeof(double *));
    for (int i = 0; i < NUM_TRAIN; i++) {
      trainInputs[i] = trainImages[i];
    }

    trainModel(myModel, trainInputs, NUM_TRAIN, trainLabels, outputs, epochs, 0.01);

    char answer, fileName[MAX_NAME_LENGTH];

    fflush(stdout);
    printf("Do you want to save the trained model? y/n (default: y): ");
    scanf(" %c", &answer);

    switch (answer) {
    case 'n':
      break;

    default:
      printf("File name: ");
      scanf(" %s", fileName);

      if (strstr(fileName, ".txt") == NULL) {
        strcat(fileName, ".txt");
      }

      saveModel(myModel, fileName, activationFunctionToUse);
      break;
    }
  }

  double **testOutputs = malloc(NUM_TEST * sizeof(double *));
  for (int i = 0; i < NUM_TEST; i++) {
    testOutputs[i] = testImages[i];
  }

  double accuracy = calculateAccuracy(myModel, testOutputs, NUM_TEST, testLabels, outputs);
  printf("\nCalculated accuracy: %.2f", accuracy * 100);
  printf("%%");

  while (1) {
    int choice;
    printf("\n\nWhat do you want to do?");
    printf("\n1 - Recognize a test image");
    printf("\n2 - Recognize a custom image (must be in the same folder as the exe)");
    printf("\n0 - Exit");
    printf("\nYour choice: ");
    scanf(" %d", &choice);

    switch (choice) {
    case 1:
      recognizeTestImage(myModel, outputs);
      break;

    case 2:
      recognizeCustomImage(myModel, outputs);
      break;

    default:
      printf("Bye!");
      return EXIT_SUCCESS;
      break;
    }
  }
}
