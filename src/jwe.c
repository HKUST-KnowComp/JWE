// Copyright 2013 Google Inc. All Rights Reserved.
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

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <locale.h>
#include <wchar.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

//Unicode range of Chinese characters
#define MIN_CHINESE 0x4E00
#define MAX_CHINESE 0x9FA5
// number of chinese characters
#define CHAR_SIZE (MAX_CHINESE - MIN_CHINESE + 1)
#define COMP_SIZE 14000
// Maximum 30 * 0.7 = 21M words in the vocabulary
const int vocab_hash_size = 30000000; 

typedef float real;

struct vocab_word{
	long long cn;    
	char *word;
	int *character, character_size;
	/*
	 * cn             :  the count of a word
	 * character[i]   : Unicode (the i-th character in the word) - MIN_CHINESE
	 * character_size : the length of the word 
	 					(not equal to the length of string due to UTF-8 encoding)
	 */
};

struct char_component{
	int *comp, comp_size;  // comp[i]  : the i -th component   comp_size, the number of components
};

struct components{
	char* comp_val;
};

char train_file[MAX_STRING], comp_file[MAX_STRING],char2comp_file[MAX_STRING];
char output_word[MAX_STRING], output_char[MAX_STRING], output_comp[MAX_STRING];
struct vocab_word *vocab;
struct char_component char2comp[CHAR_SIZE];
struct components *comp_array;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, iter = 5, num_threads = 1, min_reduce = 1;
int join_type = 1;   // 1 :  sum loss  2  : sum context
int pos_type = 1;  // 1:  use the surrounding subcomponents 2: use the target subcomponents, 3 use both
int average_sum = 1; // 1: use average operation to compose the context, 0, use sum to compose the context
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, comp_max_size = COMP_SIZE, comp_size = 0, layer1_size = 200;
long long train_words = 0, word_count_actual = 0, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1neg, *expTable, *synchar, *syncomp;
clock_t start;

int negative = 0;
const int table_size = 1e8;      //the unigram table for negative sampling
int *table;

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1, len, i;
  wchar_t wstr[MAX_STRING];
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  len = mbstowcs(wstr, word, MAX_STRING);
  for (i = 0; i < len; i++)
    if (wstr[i] < MIN_CHINESE || wstr[i] > MAX_CHINESE) {
      vocab[vocab_size - 1].character = 0;
      vocab[vocab_size - 1].character_size = 0;
      return vocab_size - 1;
    }
  vocab[vocab_size - 1].character = calloc(len, sizeof(int));
  vocab[vocab_size - 1].character_size = len;
  for (i = 0; i < len; i++)
  	vocab[vocab_size - 1].character[i] = wstr[i] - MIN_CHINESE;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab(){
	int a;
	for (a = 0; a < vocab_size; a++){
		if (vocab[a].word != NULL)
			free(vocab[a].word);
		if (vocab[a].character != NULL)
			free(vocab[a].character);
	}
	for (a = 0; a < CHAR_SIZE; a++){
		if (char2comp[a].comp != NULL)
			free(char2comp[a].comp);
	}
  for(a = 0; a < comp_size; a++){
    if(comp_array[a].comp_val != NULL)
      free(comp_array[a].comp_val);
  }
	free(vocab[vocab_size].word);
	free(vocab);
	free(comp_array);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      if (vocab[a].character != NULL) free(vocab[a].character);
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  /*
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }*/
}
// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    memcpy(vocab + b, vocab + a, sizeof(struct vocab_word));
    b++;
  } else {
    if (vocab[a].character != NULL) free(vocab[a].character);
    free(vocab[a].word);
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; //initialize vocab_hash array
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    fprintf(stderr,"ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}
// Read Component List
void ReadComponent() {
  FILE *fin;
  fin = fopen(comp_file, "rb");
  if (fin == NULL){
    fprintf(stderr,"ERROR : component file not found!\n");
    exit(1);
  }
  while(1){
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) break;
    if(comp_size + 2 >= comp_max_size){
      comp_max_size += COMP_SIZE;
      comp_array = (struct components *)realloc(comp_array,comp_max_size * sizeof(struct components));
    }
    unsigned int word_len = strlen(word) + 1;
    comp_array[comp_size].comp_val = (char *)calloc(word_len, sizeof(char));
    strcpy(comp_array[comp_size].comp_val, word);
    comp_size++; 
  }
  fclose(fin);
}
// find the index of a component in the component array
int GetCompIndex(char *component){
  for(int i = 0; i < comp_size; i++){
    if(strcmp(component, comp_array[i].comp_val) == 0)
      return i;
  }
  return -1;
}
//Read char2comp and component array from file
void LearnCharComponentsFromFile() {  
	FILE *fin;
	fin = fopen(char2comp_file,"rb");
	if(fin == NULL){
		fprintf(stderr,"ERROR: char2component file not found!\n");
		exit(1);
	}
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
	while ((read = getline(&line, &len, fin)) != -1){
    //parse the line , get characters and its components
    int num = (strlen(line) - 2) / 3 - 1;
    char *save_ptr;
    char *pch = strtok_r(line, " \n",&save_ptr);
    wchar_t wstr[MAX_STRING];
    unsigned int wlen = mbstowcs(wstr, pch, MAX_STRING);
    int id = wstr[0] - MIN_CHINESE;
    char2comp[id].comp_size = num;
    char2comp[id].comp = calloc(char2comp[id].comp_size, sizeof(int));           //
    int tmp_cnt = 0;
    pch = strtok_r(NULL," \n",&save_ptr);
    while(pch != NULL){
      int pch_index = GetCompIndex(pch);
      if(pch_index != -1)
        char2comp[id].comp[tmp_cnt++] = pch_index;
      pch = strtok_r(NULL," \n",&save_ptr);
    }
	}
  if(line)
    free(line);
	fclose(fin);
}

void InitNet(){
  long long a, b;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&synchar, 128, (long long)CHAR_SIZE * layer1_size * sizeof(real));
  if (synchar == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&syncomp, 128, (long long)comp_size * layer1_size * sizeof(real));
  if (syncomp == NULL) {printf("Memory allocation failed\n"); exit(1);}
  //Initialize the weights
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
    syn1neg[a * layer1_size + b] = 0;
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
    syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  for (b = 0; b < layer1_size; b++) for (a = 0; a < CHAR_SIZE; a++)
    synchar[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  for (b = 0; b < layer1_size; b++) for (a = 0; a < comp_size; a++)
    syncomp[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
}

void DestroyNet(){
  if (syn0 != NULL){
    free(syn0);
  }
  if (syn1neg != NULL){
    free(syn1neg);
  }
  if (synchar != NULL){
    free(synchar);
  }
  if (syncomp != NULL){
    free(syncomp);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, c, d, e, char_id, comp_id, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2,  target, label, local_iter = iter;
  long long *char_id_list = calloc(MAX_SENTENCE_LENGTH, sizeof(long long));
  long long *comp_id_list = calloc(MAX_SENTENCE_LENGTH, sizeof(long long));
  int char_list_cnt = 0, comp_list_cnt = 0;
  unsigned long long next_random = (long long)id;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1_grad = (real *)calloc(layer1_size,sizeof(real));    
  real *neuchar = (real *)calloc(layer1_size,sizeof(real));
  real *neuchar_grad = (real *)calloc(layer1_size,sizeof(real));
  real *neucomp = (real *)calloc(layer1_size,sizeof(real));
  real *neucomp_grad = (real *)calloc(layer1_size,sizeof(real));
  FILE *fi = fopen(train_file,"rb");
  if (fi == NULL){
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi,file_size / (long long) num_threads * (long long)id, SEEK_SET);
  while (1) {   
    //decay learning rate and print training progress
    if(word_count - last_word_count > 10000){
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
           word_count_actual / (real)(iter * train_words + 1) * 100,
           word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    // read a word sentence 
    if (sentence_length == 0){
      while (1){
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break; 
        // the subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    //if (feof(fi)) break;
    //if (word_count > train_words / num_threads) break;
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    // train the cbow model
    // before forward backward propagation, initialize the neurons and gradients to 0
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1_grad[c] = 0;
    for (c = 0; c < layer1_size; c++) neuchar[c] = 0;
    for (c = 0; c < layer1_size; c++) neuchar_grad[c] = 0;
    for (c = 0; c < layer1_size; c++) neucomp[c] = 0;
    for (c = 0; c < layer1_size; c++) neucomp_grad[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;  //[0, window-1]
    char_list_cnt = 0;
    comp_list_cnt = 0;
    int cw = 0;
    // in -> hidden         get contex sum vector 
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
      c = sentence_position - window + a;
      if (c < 0) continue;
      if (c >= sentence_length) continue;
      last_word = sen[c];
      if (last_word == -1) continue;
      for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      for (c = 0; c < vocab[last_word].character_size; c++) {
        char_id = vocab[last_word].character[c];
        char_id_list[char_list_cnt++] = char_id;
        for (d = 0; d < layer1_size; d++) neuchar[d] += synchar[d + char_id * layer1_size];
        //use the surrounding characters' component information
        if (pos_type == 1 || pos_type == 3) {
          for (d = 0; d < char2comp[char_id].comp_size; d++) {
            comp_id = char2comp[char_id].comp[d];
            comp_id_list[comp_list_cnt++] = comp_id;
            for (e = 0; e < layer1_size; e++) neucomp[e] += syncomp[e + comp_id * layer1_size];
          }
        }
      }
      cw++;
    }
    // use the target character's component information
    if (pos_type == 2 || pos_type == 3) {
      last_word = sen[sentence_position];
      for (c = 0; c < vocab[last_word].character_size; c++) {
        char_id = vocab[last_word].character[c];
        for (d = 0; d < char2comp[char_id].comp_size; d++) {
          comp_id = char2comp[char_id].comp[d];
          comp_id_list[comp_list_cnt++] = comp_id;
          for (e = 0; e < layer1_size; e++) neucomp[e] += syncomp[e + comp_id * layer1_size];
        }
      }
    }
    if (cw) {
      if (average_sum == 1) {       // the context is represented by the average of the surrounding vectors
        for (c = 0; c < layer1_size; c++) {
          neu1[c] /= cw;
          if (char_list_cnt > 0)
            neuchar[c] /= char_list_cnt;
          if (comp_list_cnt > 0)
            neucomp[c] /= comp_list_cnt;
        }
      }
      // negative sampling
      for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        }
        else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;  // if sample "</s>", randomly resample
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        // back propagate      output  -->   hidden
        if (join_type == 1) {    // sum loss composition model
          real f1 = 0, f2 = 0, f3 = 0, g1 = 0, g2 = 0, g3 = 0;
          for (c = 0; c < layer1_size; c++) {
            f1 += neu1[c] * syn1neg[c + l2];
            f2 += neuchar[c] * syn1neg[c + l2];
            f3 += neucomp[c] * syn1neg[c + l2];
          }
          if (f1 > MAX_EXP)
            g1 = (label - 1) * alpha;
          else if (f1 < -MAX_EXP)
            g1 = (label - 0) * alpha;
          else
            g1 = (label - expTable[(int)((f1 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          if (f2 > MAX_EXP)
            g2 = (label - 1) * alpha;
          else if (f2 < -MAX_EXP)
            g2 = (label - 0) * alpha;
          else
            g2 = (label - expTable[(int)((f2 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          if (f3 > MAX_EXP)
            g3 = (label - 1) * alpha;
          else if (f3 < -MAX_EXP)
            g3 = (label - 0) * alpha;
          else
            g3 = (label - expTable[(int)((f3 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          // compute the gradients of neurons
          for (c = 0; c < layer1_size; c++) {
            neu1_grad[c] += g1 * syn1neg[c + l2];
            neuchar_grad[c] += g2 * syn1neg[c + l2];
            neucomp_grad[c] += g3 * syn1neg[c + l2];
          }
          //update syn1neg
          for (c = 0; c < layer1_size; c++)
            syn1neg[c + l2] += g1 * neu1[c] + g2 * neuchar[c] + g3 * neucomp[c];
        }
        else if (join_type == 2) { // average context composition model  
          real f = 0, g = 0;
          for (c = 0; c < layer1_size; c++)
            f += (neu1[c] + neuchar[c] + neucomp[c]) * syn1neg[c + l2];
          if (f > MAX_EXP)
            g = (label - 1) * alpha;
          else if (f < -MAX_EXP)
            g = (label - 0) * alpha;
          else
            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) {
            neu1_grad[c] += g * syn1neg[c + l2];
            neucomp_grad[c] += g * syn1neg[c + l2];
            neuchar_grad[c] += g * syn1neg[c + l2];
          }
          for (c = 0; c < layer1_size; c++)
            syn1neg[c + l2] += g * (neu1[c] + neuchar[c] + neucomp[c]);
        }
      }
      // back propagate   hidden -> input
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++){
            if (average_sum == 1)
              syn0[c + last_word * layer1_size] += neu1_grad[c] / cw;
            else
              syn0[c + last_word * layer1_size] += neu1_grad[c];
          }
        }
      for (a = 0; a < char_list_cnt; a++) {
        char_id = char_id_list[a];
        for (c = 0; c < layer1_size; c++){
          if (average_sum == 1)
            synchar[c + char_id * layer1_size] += neuchar_grad[c] / char_list_cnt;
          else
            synchar[c + char_id * layer1_size] += neuchar_grad[c];
        }
      }
      for (a = 0; a < comp_list_cnt; a++) {
        comp_id = comp_id_list[a];
        for (c = 0; c < layer1_size; c++){
          if (average_sum == 1)
            syncomp[c + comp_id * layer1_size] += neucomp_grad[c] / comp_list_cnt;
          else
            syncomp[c + comp_id * layer1_size] += neucomp_grad[c];
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length){
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1_grad);
  pthread_exit(NULL);
}
void TrainModel(){
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL){
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s \n", train_file);
  starting_alpha = alpha;
  LearnVocabFromTrainFile();
  ReadComponent();
  LearnCharComponentsFromFile();
  if (output_word[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  // save the word vectors
  fo = fopen(output_word, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_word);
    exit(1);
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL)
      fprintf(fo, "%s ", vocab[a].word);
    if (binary)
      for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else
      for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  wchar_t ch[10];
  if (strlen(output_char)){
    fo = fopen(output_char, "wb");
    if (fo == NULL){
      fprintf(stderr, "Cannot open %s: permission denied\n", output_char);
    }
    fprintf(fo, "%lld %lld\n", CHAR_SIZE, layer1_size);
    for (a = 0; a < CHAR_SIZE; a++){
      ch[0] = MIN_CHINESE + a;
      ch[1] = 0;
      fprintf(fo, "%ls\t", ch);
      if (binary)
        for (b = 0; b < layer1_size; b++) fwrite(&synchar[a * layer1_size + b], sizeof(real), 1, fo);
      else
        for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", synchar[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
  if (strlen(output_comp)){
    fo = fopen(output_comp, "wb");
    if (fo == NULL){
      fprintf(stderr, "Cannot open %s: permission denied\n", output_comp);
    }
    fprintf(fo, "%lld %lld\n", comp_size, layer1_size);
    for(a = 0; a < comp_size; a++){
      fprintf(fo, "%s ", comp_array[a]);
      if (binary)
        for (b = 0; b < layer1_size; b++) fwrite(&syncomp[a * layer1_size + b], sizeof(real), 1, fo);
      else
        for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syncomp[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
  free(table);
  free(pt);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  setlocale(LC_ALL, "en_US.UTF-8");
  if(argc == 1){
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
      printf("Options:\n");
      printf("Parameters for training:\n");
      printf("\t-train <file>\n");
      printf("\t\tUse text data from <file> to train the model\n");
      printf("\t-output-word <file>\n");
      printf("\t\tUse <file> to save the resulting word vectors\n");
      printf("\t-output-char <file>\n");
      printf("\t\tUse <file> to save the resulting character vectors\n");
      printf("\t-output-comp <file>\n");
      printf("\t\tUse <file> to save the resulting component vectors\n");
      printf("\t-size <int>\n");
      printf("\t\tSet size of word vectors; default is 200\n");
      printf("\t-window <int>\n");
      printf("\t\tSet max skip length between words; default is 5\n");
      printf("\t-sample <float>\n");
      printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
      printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
      printf("\t-negative <int>\n");
      printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
      printf("\t-iter <int>\n");
      printf("\t\tRun more training iterations (default 5)\n");
      printf("\t-threads <int>\n");
      printf("\t\tUse <int> threads (default 1)\n");
      printf("\t-min-count <int>\n");
      printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
      printf("\t-alpha <float>\n");
      printf("\t\tSet the starting learning rate; default is 0.025\n");
      printf("\t-debug <int>\n");
      printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
      printf("\t-binary <int>\n");
      printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
      printf("\t-comp <file>\n");
      printf("\t\tUse component list from <file>\n");
      printf("\t-char2comp <file>\n");
      printf("\t\tObtain the mapping between characters and their components from <file>\n");
      printf("\t-join-type <int>\n");
      printf("\t\t The type of methods combining subwords (default = 1: sum loss, 2 : sum context)\n");
      printf("\t-pos-type <int>\n");
      printf("\t\t The type of subcomponents' positon (default = 1: use the components of surrounding words, 2: use the components of the target word, 3: use both)\n");
      printf("\t-average-sum <int>\n");
      printf("\t\t The type of compositional method for context (average_sumdefault = 1 use average operation, 0 use sum operation)\n");
      printf("\nExamples:\n");
      printf("./jwe -train wiki_process.txt -output-word word_vec -output-char char_vec -output-comp comp_vec -size 200 -window 5 -sample 1e-4 -negative 10 -iter 100 -threads 8 -min-count 5 -alpha 0.025 -binary 0 -comp ../subcharacter/comp.txt -char2comp ../subcharacter/char2comp.txt -join-type 1 -pos-type 1 -average-sum 1\n\n");
      return 0;
  }
  output_word[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output-word", argc, argv)) > 0) strcpy(output_word, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-char", argc, argv)) > 0) strcpy(output_char, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-comp", argc, argv)) > 0) strcpy(output_comp, argv[i + 1]);
  if ((i = ArgPos((char *)"-comp", argc, argv)) > 0) strcpy(comp_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-char2comp", argc, argv)) > 0) strcpy(char2comp_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-join-type", argc, argv)) > 0) join_type = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-pos-type", argc, argv)) > 0) pos_type = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-average-sum", argc, argv)) > 0) average_sum = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  comp_array = (struct components *)calloc(comp_max_size, sizeof(struct components));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;
}
