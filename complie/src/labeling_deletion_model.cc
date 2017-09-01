#include "labeling_deletion_model.h"
#include "logging.h"
#include <boost/assert.hpp>


LabelingDeletionModel::LabelingDeletionModel(
  const std::vector<std::string>& id_to_label_,
  bool enable_tran_constrain)
  : enable(enable_tran_constrain), id_to_label(id_to_label_) {
}


void LabelingDeletionModel::get_possible_labels(std::vector<unsigned>& possible_labels) {
  // Get possible labels for starting point.
  for (unsigned i = 0; i < id_to_label.size(); ++i) {
    if (enable) {
      const std::string& str = id_to_label[i];
      if (str[0] == 'O' || str[0] == 'o' || str[0] == 'B' || str[0] == 'b') {
        possible_labels.push_back(i);
      }
    } else {
      possible_labels.push_back(i);
    }
  }
}


void LabelingDeletionModel::get_possible_labels(unsigned prev,
  std::vector<unsigned>& possible_labels) {
  if (enable) {
    BOOST_ASSERT_MSG(prev < id_to_label.size(), "Previous label id not in range.");
    const std::string& prev_str = id_to_label[prev];

    char prev_ch = prev_str[0];
    if (prev_ch == 'O' || prev_ch == 'o' || prev_ch == 'S' || prev_ch == 's' || prev_ch == 'E' || prev_ch == 'e') {
      for (unsigned i = 0; i < id_to_label.size(); ++i) {
        char ch = id_to_label[i][0];
        if (ch == 'I' || ch == 'i' || ch == 'E' || ch == 'e') {
          continue;
        }
        possible_labels.push_back(i);
      }
    } else if (prev_ch == 'B' || prev_ch == 'b' || prev_ch == 'I' || prev_ch == 'i') {
      for (unsigned i = 0; i < id_to_label.size(); ++i) {
        char ch = id_to_label[i][0];
        if (ch == 'B' || ch == 'b' || ch == 'S' || ch == 's' || ch == 'O' || ch == 'o') {
          continue;
        }
        possible_labels.push_back(i);
      }
    } else {
      _ERROR << "unknown previous tag";
    }
  } else {
    for (unsigned i = 0; i < id_to_label.size(); ++i) {
      possible_labels.push_back(i);
    }
  }
}


unsigned LabelingDeletionModel::get_best_scored_label(const std::vector<float>& scores,
  const std::vector<unsigned>& possible_labels) {
  float best_score = scores[possible_labels[0]];
  unsigned best_lid = 0;
  for (unsigned j = 1; j < possible_labels.size(); ++j) {
    if (best_score < scores[possible_labels[j]]) {
      best_score = scores[possible_labels[j]];
      best_lid = possible_labels[j];
    }
  }
  return best_lid;
}

