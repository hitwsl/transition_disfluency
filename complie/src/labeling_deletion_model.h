#ifndef __LABEL_DELETION_MODEL_H__
#define __LABEL_DELETION_MODEL_H__

#include <iostream>
#include <vector>

struct LabelingDeletionModel {
  bool enable; // Whether enable the transition constrain.
  const std::vector<std::string>& id_to_label; // recording the mapping from id to label

  LabelingDeletionModel(const std::vector<std::string>& id_to_label,
    bool enable_tran_constrain);

  void get_possible_labels(std::vector<unsigned>& possible_labels);
  void get_possible_labels(unsigned prev, std::vector<unsigned>& possible_labels);
  unsigned get_best_scored_label(const std::vector<float>& scores,
    const std::vector<unsigned>& possible_labels);
};


#endif  //  end for __LABEL_DELETION_MODEL_H__