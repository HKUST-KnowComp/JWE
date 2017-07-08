//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x

const long long max_size = 2000;         // max length of strings
const long long N = 1;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries



int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], st[100][max_size];
  char analogy_file_name[max_size];
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;
  char *vocab;
  int total[3], hit[3]; // three categories: capital state family
  int category;
  int bestidx;
  int debug;

  if (argc < 3) {
    printf("Usage: ./word-analogy <FILE> <AFILE>\nwhere FILE contains word projections in the BINARY FORMAT and AFILE contains analogy\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  strcpy(analogy_file_name, argv[2]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s", &vocab[b * max_w]);
    if (b < 10) {
	    //printf("%s\n", &vocab[b * max_w]);
    }
    for (a = 0; a < size; a++) fscanf(f, "%f", &M[a + b * size]);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);

  for (b = 0; b < 3; b++) {
    total[b] = 0;
    hit[b] = 0;
  }
  f = fopen(analogy_file_name, "rb");
  category = -1;
  debug = 10;
  while (!feof(f)) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    a = 0;
    while (1) {
      st1[a] = fgetc(f);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (st1[0] == ':') {
      category++;
      continue;
    }
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    if (debug > 0) {
	    debug--;
	    //printf("%s %s %s %s\n", st[0], st[1], st[2], st[3]);
    }
    //cn++;
    for (a = 0; a < cn + 1; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = 0;
      bi[a] = b;
      // printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == 0) {
        //printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == 0) continue;
    // printf("\n                                              Word              Distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      if (c == bi[0]) continue;
      if (c == bi[1]) continue;
      if (c == bi[2]) continue;
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          bestidx = c;
          break;
        }
      }
    }
    //for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    total[category]++;
    if (bestidx == bi[3]) hit[category]++;
    if (debug < 10) {
	    //printf("%d %d\n", bestidx, bi[3]);
    }
  }
  printf("capital: %d/%d, state: %d/%d, family: %d/%d\n", hit[0], total[0], hit[1], total[1], hit[2], total[2]);
  return 0;
}
