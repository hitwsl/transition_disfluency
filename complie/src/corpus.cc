#include "utils.h"
#include "logging.h"
#include "corpus.h"
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>


Corpus::Corpus()
  : max_word(0), max_postag(0), max_char(0) {
}







void Corpus::load_training_data(const std::string& train_filename, const std::string& train_lm_filename, unsigned gen_feature_dim) {
  std::ifstream train_file(train_filename);
  BOOST_ASSERT_MSG(train_file, "Failed to open training file.");
  std::string line;
  word_to_id[Corpus::BAD0] = 0; id_to_word[0] = Corpus::BAD0;
  word_to_id[Corpus::UNK] = 1;  id_to_word[1] = Corpus::UNK;
  word_to_id["<s>"] = 2;  id_to_word[2] = "<s>";
  word_to_id["</s>"] = 3;  id_to_word[3] = "</s>";


  BOOST_ASSERT_MSG(max_word == 0, "max_word is set before loading training data!");
  max_word = 4;

  postag_to_id[Corpus::BAD0] = 0; id_to_postag[0] = Corpus::BAD0;
  BOOST_ASSERT_MSG(max_postag == 0, "max_postag is set before loading training data!");
  max_postag = 1;

  char_to_id[Corpus::BAD0] = 1; id_to_char[1] = Corpus::BAD0;
  max_char = 1;

  std::vector<unsigned> current_sentence;
  std::vector<unsigned> current_postag;
  std::vector<unsigned> current_label;
  std::vector<std::vector<float>> current_gen_feature;
  std::string temp;
  std::vector<float> v(gen_feature_dim, 0);


  unsigned sid = 0;
  while (std::getline(train_file, line)) {
    replace_string_in_place(line, "-RRB-", "_RRB_");
    replace_string_in_place(line, "-LRB-", "_LRB_");

    if (line.empty()) {
      if (current_sentence.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      train_sentences[sid] = current_sentence;
      train_postags[sid] = current_postag;
      train_labels[sid] = current_label;
      train_general_feature[sid] = current_gen_feature;

      sid ++;
      n_train = sid;
      current_sentence.clear();
      current_postag.clear();
      current_label.clear();
      current_gen_feature.clear();
    } else {
      boost::algorithm::trim(line);
      std::vector<std::string> items;
      boost::algorithm::split(items, line, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() >= 4, "Ill formated CoNLL data");
      const std::string& word = items[0];
      const std::string& pos = items[1];
      const std::string& label = items[3];

      add(pos, max_postag, postag_to_id, id_to_postag);
      add(word, max_word, word_to_id, id_to_word);

      unsigned lid;
      bool found = find(label, id_to_label, lid);

      if (!found) {
        id_to_label.push_back(label);
        lid = id_to_label.size() - 1;
      }

      current_sentence.push_back(word_to_id[word]);
      current_postag.push_back(postag_to_id[pos]);
      current_label.push_back(lid);

      std::istringstream iss(line);
      for(unsigned i = 0; i < 4; i++){iss >> temp;}
      for(unsigned i = 0; i < gen_feature_dim; i++){iss >> v[i];
      }

      current_gen_feature.push_back(v);

    }
  }

  if (current_sentence.size() > 0) {
    train_sentences[sid] = current_sentence;
    train_postags[sid] = current_postag;
    train_labels[sid] = current_label;
    train_general_feature[sid] = current_gen_feature;
    sid++;
    n_train = sid;
    current_sentence.clear();
    current_postag.clear();
    current_label.clear();
    current_gen_feature.clear();
  }

  train_file.close();
  _INFO << "finish loading training data.";
  n_labels = id_to_label.size();
  stat();

  std::ifstream train_file_lm(train_lm_filename);
  BOOST_ASSERT_MSG(train_file_lm, "Failed to open training LM file.");
  std::string line_lm;

  std::vector<unsigned> current_sentence_lm;

  unsigned sid_lm = 0;
  while (std::getline(train_file_lm, line_lm)) {

    if (line_lm.empty()) {
      if (current_sentence_lm.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      //train_sentences_lm[sid_lm] = current_sentence_lm;
      train_sentences_lm.push_back(current_sentence_lm);
      sid_lm ++;
      n_train_lm = sid_lm;
      current_sentence_lm.clear();

    } else {
      boost::algorithm::trim(line_lm);
      std::vector<std::string> items;
      boost::algorithm::split(items, line_lm, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() == 1, "Ill formated LM data");
      const std::string& word = items[0];
      add(word, max_word, word_to_id, id_to_word);
      current_sentence_lm.push_back(word_to_id[word]);

    }
  }

  if (current_sentence_lm.size() > 0) {
   // train_sentences_lm[sid_lm] = current_sentence_lm;
    train_sentences_lm.push_back(current_sentence_lm);

    n_train_lm = sid_lm + 1;
    current_sentence_lm.clear();
  }
//  CompareLen comp;
//  sort(train_sentences_lm.begin(), train_sentences_lm.end(), comp);

  train_file_lm.close();
  _INFO << "finish loading training_lm data.";
  stat_lm();

}


void Corpus::load_devel_data(const std::string& devel_filename, const std::string& devel_lm_filename, unsigned gen_feature_dim) {
  std::ifstream devel_file(devel_filename);
  std::string line;

  BOOST_ASSERT_MSG(max_word > 3, "max_word is not set before loading development data!");
  BOOST_ASSERT_MSG(max_postag > 1, "max_postag is not set before loading development data!");

  std::vector<unsigned> current_sentence;
  std::vector<std::string> current_sentence_str;
  std::vector<unsigned> current_postag;
  std::vector<std::vector<float>> current_gen_feature;
  std::vector<unsigned> current_label;
  std::string temp;
  std::vector<float> v(gen_feature_dim, 0);

  unsigned sid = 0;
  while (std::getline(devel_file, line)) {
    replace_string_in_place(line, "-RRB-", "_RRB_");
    replace_string_in_place(line, "-LRB-", "_LRB_");

    if (line.empty()) {
      if (current_sentence.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      devel_sentences[sid] = current_sentence;
      devel_sentences_str[sid] = current_sentence_str;
      devel_postags[sid] = current_postag;
      devel_general_feature[sid] = current_gen_feature;
      devel_labels[sid] = current_label;

      sid++;
      n_devel = sid;
      current_sentence.clear();
      current_sentence_str.clear();
      current_postag.clear();
      current_gen_feature.clear();
      current_label.clear();
    } else {
      boost::algorithm::trim(line);
      std::vector<std::string> items;
      boost::algorithm::split(items, line, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() >= 4, "Ill formated CoNLL data");
      const std::string& word = items[0];
      const std::string& pos = items[1];
      const std::string& label = items[3];

      current_sentence_str.push_back(word);

      unsigned payload = 0;
      if (!find(pos, postag_to_id, payload)) {
        BOOST_ASSERT_MSG(false, "Unknow postag in development data.");
      } else {
        current_postag.push_back(payload);
      }

      if (!find(word, word_to_id, payload)) {
        find(Corpus::UNK, word_to_id, payload);
        current_sentence.push_back(payload);
      } else {
        current_sentence.push_back(payload);
      }

      bool found = find(label, id_to_label, payload);
      if (!found) {
        BOOST_ASSERT_MSG(false, "Unknow label in development data.");
      }

      current_label.push_back(payload);
      std::istringstream iss(line);
      for(unsigned i = 0; i < 4; i++){iss >> temp;}
      for(unsigned i = 0; i < gen_feature_dim; i++){iss >> v[i];}
      current_gen_feature.push_back(v);

    }
  }

  if (current_sentence.size() > 0) {
    devel_sentences[sid] = current_sentence;
    devel_sentences_str[sid] = current_sentence_str;
    devel_postags[sid] = current_postag;
    devel_labels[sid] = current_label;
    devel_general_feature[sid] = current_gen_feature;
    n_devel = sid + 1;
  }

  devel_file.close();
  _INFO << "finish load development data.";






  std::ifstream devel_file_lm(devel_lm_filename);
  std::string line_lm;


  std::vector<unsigned> current_sentence_lm;


  unsigned sid_lm = 0;
  while (std::getline(devel_file_lm, line_lm)) {

    if (line_lm.empty()) {
      if (current_sentence_lm.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      devel_sentences_lm.push_back(current_sentence_lm);
      sid_lm++;
      n_devel_lm = sid_lm;
      current_sentence_lm.clear();
    } else {
      boost::algorithm::trim(line_lm);
      std::vector<std::string> items;
      boost::algorithm::split(items, line_lm, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() == 1, "Ill formated CoNLL data");
      const std::string& word = items[0];
      unsigned payload = 0;

      if (!find(word, word_to_id, payload)) {
        find(Corpus::UNK, word_to_id, payload);
        current_sentence_lm.push_back(payload);
      } else {
        current_sentence_lm.push_back(payload);
      }
    }
  }

  if (current_sentence_lm.size() > 0) {
    devel_sentences_lm.push_back(current_sentence_lm);

    n_devel_lm = sid_lm + 1;
  }
  devel_file_lm.close();
  _INFO << "finish load LM development data.";

}


void Corpus::stat() const {
  _INFO << "action indexing ...";
  for (auto l : id_to_label) {
    _INFO << l;
  }
  _INFO << "number of labels: " << id_to_label.size();
  _INFO << "max id of words: " << max_word;
  _INFO << "max id of postags: " << max_postag;

  _INFO << "postag indexing ...";
  for (auto& p : id_to_postag) { _INFO << p.first << " : " << p.second; }
}


void Corpus::stat_lm() const {
  _INFO << "number of training lm sentence: " << n_train_lm;
}



unsigned Corpus::get_or_add_word(const std::string& word) {
  unsigned payload;
  if (!find(word, word_to_id, payload)) {
    add(word, max_word, word_to_id, id_to_word);
    return word_to_id[word];
  }
  return payload;
}

void Corpus::get_vocabulary_and_singletons(std::set<unsigned>& vocabulary,
  std::set<unsigned>& singletons) {
  std::map<unsigned, unsigned> counter;
  for (auto& payload : train_sentences) {
    for (auto& word : payload.second) {
      vocabulary.insert(word);
      ++counter[word];
    }
  }
  for (auto& payload : counter) {
    if (payload.second == 1) { singletons.insert(payload.first); }
  }
}
