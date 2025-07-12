/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C
*/

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels.idx1-ubyte"

#define SIDE_SIZE 28
#define SIZE SIDE_SIZE *SIDE_SIZE
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double trainImages[NUM_TRAIN][SIZE];
double testImages[NUM_TEST][SIZE];
int trainLabels[NUM_TRAIN];
int testLabels[NUM_TEST];

void printImage(double *img) {
  float h = 0.0f;
  float s = 0.0f;

  for (int i = 0; i < SIZE; i++) {
    int l = img[i] * MAX_BRIGHTNESS;
    printf("\033[38;2;%d;%d;%dm", l, l, l);

    printf("██");

    if ((i + 1) % SIDE_SIZE == 0)
      printf("\n");
  }

  printf("\33[0m");
}

void FlipLong(unsigned char *ptr) {
  register unsigned char val;

  // Swap 1st and 4th bytes
  val = *(ptr);
  *(ptr) = *(ptr + 3);
  *(ptr + 3) = val;

  // Swap 2nd and 3rd bytes
  ptr += 1;
  val = *(ptr);
  *(ptr) = *(ptr + 1);
  *(ptr + 1) = val;
}

void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n, unsigned char data_char[][arr_n], int info_arr[]) {
  int i, j, k, fd;
  unsigned char *ptr;

  if ((fd = open(file_path, O_RDONLY)) == -1) {
    fprintf(stderr, "couldn't open image file");
    exit(-1);
  }

  read(fd, info_arr, len_info * sizeof(int));

  // read-in information about size of data
  for (i = 0; i < len_info; i++) {
    ptr = (unsigned char *)(info_arr + i);
    FlipLong(ptr);
    ptr = ptr + sizeof(int);
  }

  // read-in mnist numbers (pixels|labels)
  for (i = 0; i < num_data; i++) {
    read(fd, data_char[i], arr_n * sizeof(unsigned char));
  }

  close(fd);
}

void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE]) {
  int i, j;
  for (i = 0; i < num_data; i++)
    for (j = 0; j < SIZE; j++)
      data_image[i][j] = (double)data_image_char[i][j] / 255.0;
}

void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[]) {
  int i;
  for (i = 0; i < num_data; i++)
    data_label[i] = (int)data_label_char[i][0];
}

void load_mnist() {
  read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE, train_image_char, info_image);
  image_char2double(NUM_TRAIN, train_image_char, trainImages);

  read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, test_image_char, info_image);
  image_char2double(NUM_TEST, test_image_char, testImages);

  read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
  label_char2int(NUM_TRAIN, train_label_char, trainLabels);

  read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
  label_char2int(NUM_TEST, test_label_char, testLabels);
}

void print_mnist_pixel(double data_image[][SIZE], int num_data) {
  int i, j;
  for (i = 0; i < num_data; i++) {
    printf("image %d/%d\n", i + 1, num_data);
    for (j = 0; j < SIZE; j++) {
      printf("%1.1f ", data_image[i][j]);
      if ((j + 1) % 28 == 0)
        putchar('\n');
    }
    putchar('\n');
  }
}

void print_mnist_label(int data_label[], int num_data) {
  int i;
  if (num_data == NUM_TRAIN)
    for (i = 0; i < num_data; i++)
      printf("train_label[%d]: %d\n", i, trainLabels[i]);
  else
    for (i = 0; i < num_data; i++)
      printf("test_label[%d]: %d\n", i, testLabels[i]);
}
