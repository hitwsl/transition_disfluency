#ifndef __CPYPDICT_H__
#define __CPYPDICT_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <vector>
#include <functional>
#include "utils.h"

struct Corpus: public CorpusI {
  std::map<int, std::vector<unsigned>> train_labels;
  std::map<int, std::vector<unsigned>> train_sentences;
  //std::map<int, std::vector<unsigned>> train_sentences_lm;
  std::vector<std::vector<unsigned>> train_sentences_lm;
  std::map<int, std::vector<unsigned>> train_postags;
  std::map<int, std::vector<std::vector<float>>> train_general_feature;

  std::map<int, std::vector<unsigned>> devel_labels;
  std::map<int, std::vector<unsigned>> predict_labels;
  std::map<int, std::vector<unsigned>> devel_sentences;
  //std::map<int, std::vector<unsigned>> devel_sentences_lm;
  std::vector<std::vector<unsigned>> devel_sentences_lm;
  std::map<int, std::vector<unsigned>> devel_postags;
  std::map<int, std::vector<std::string>> devel_sentences_str;
  std::map<int, std::vector<std::vector<float>>> devel_general_feature;

  unsigned n_train; /* number of sentences in the training data */
  unsigned n_train_lm; /* number of sentences in the training data */
  unsigned n_devel; /* number of sentences in the development data */
  unsigned n_devel_lm; /* number of sentences in the development data */

  unsigned n_labels;

  unsigned max_word;
  StringToIdMap word_to_id;
  IdToStringMap id_to_word;

  unsigned max_postag;
  StringToIdMap postag_to_id;
  IdToStringMap id_to_postag;

  unsigned max_char;
  StringToIdMap char_to_id;
  IdToStringMap id_to_char;

  std::vector<std::string> id_to_label;

  Corpus();

  void load_training_data(const std::string& train_filename, const std::string& train_lm_filename, unsigned gen_feature_dim);
  void load_devel_data(const std::string& devel_filename, const std::string& devel_lm_filename, unsigned gen_feature_dim);
  void stat() const;
  void stat_lm() const;
  unsigned get_or_add_word(const std::string& word);
  void get_vocabulary_and_singletons(std::set<unsigned>& vocabulary, std::set<unsigned>& singletons);
};

#endif  //  end for __CPYPDICT_H__
