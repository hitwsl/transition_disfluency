#include "layer.h"

SymbolEmbedding::SymbolEmbedding(cnn::Model *m, unsigned n, unsigned dim)
        :
        p_labels(m->add_lookup_parameters(n, {dim, 1})) {
}

cnn::expr::Expression SymbolEmbedding::embed(cnn::ComputationGraph *cg, unsigned label_id) {
    return cnn::expr::lookup(*cg, p_labels, label_id);
}

ConstSymbolEmbedding::ConstSymbolEmbedding(cnn::Model *m, unsigned n, unsigned dim)
        :
        p_labels(m->add_lookup_parameters(n, {dim, 1})) {
}

cnn::expr::Expression ConstSymbolEmbedding::embed(cnn::ComputationGraph *cg, unsigned label_id) {
    return cnn::expr::const_lookup(*cg, p_labels, label_id);
}

BinnedDistanceEmbedding::BinnedDistanceEmbedding(cnn::Model *m, unsigned dim, unsigned n_bins)
        :
        p_e(m->add_lookup_parameters(n_bins * 2, {dim, 1})),
        max_bin(n_bins - 1) {
    BOOST_ASSERT(n_bins > 0);
}

cnn::expr::Expression BinnedDistanceEmbedding::embed(cnn::ComputationGraph *cg, int distance) {
    unsigned base = (distance < 0 ? max_bin : 0);
    if (distance) {
        distance = static_cast<unsigned>(log(distance < 0 ? -distance : distance) / log(1.6f)) + 1;
    }
    if (distance > max_bin) {
        distance = max_bin;
    }
    return cnn::expr::lookup(*cg, p_e, distance + base);
}

BinnedDurationEmbedding::BinnedDurationEmbedding(cnn::Model *m, unsigned dim, unsigned n_bins)
        : p_e(m->add_lookup_parameters(n_bins, {dim, 1})),
          max_bin(n_bins - 1) {
    BOOST_ASSERT(n_bins > 0);
}

cnn::expr::Expression BinnedDurationEmbedding::embed(cnn::ComputationGraph *cg, unsigned dur) {
    if (dur) {
        dur = static_cast<unsigned>(log(dur) / log(1.6f)) + 1;
    }
    if (dur > max_bin) {
        dur = max_bin;
    }
    return cnn::expr::lookup(*cg, p_e, dur);
}

StaticInputLayer::StaticInputLayer(cnn::Model *model, unsigned dim_gen,
                                   unsigned size_word, unsigned dim_word,
                                   unsigned size_postag, unsigned dim_postag,
                                   unsigned size_pretrained_word, unsigned dim_pretrained_word,
                                   unsigned dim_output,
                                   const std::unordered_map<unsigned, std::vector<float>> &pretrained) :
        p_w(nullptr), p_p(nullptr), p_t(nullptr),
        p_ib(nullptr), p_w2l(nullptr), p_p2l(nullptr), p_t2l(nullptr), p_g2l(nullptr),
        use_word(true), use_postag(true), use_pretrained_word(true) {

    p_ib = model->add_parameters({dim_output, 1});
    if (dim_word == 0) {
        std::cerr << "Word dim should be greater than 0." << std::endl;
        std::cerr << "Fine-tuned word embedding is inactivated." << std::endl;
        use_word = false;
    } else {
        p_w = model->add_lookup_parameters(size_word, {dim_word, 1});
        p_w2l = model->add_parameters({dim_output, dim_word});
    }

    if (dim_postag == 0) {
        std::cerr << "Postag dim should be greater than 0." << std::endl;
        std::cerr << "Fine-tuned postag embedding is inactivated." << std::endl;
        use_postag = false;
    } else {
        p_p = model->add_lookup_parameters(size_postag, {dim_postag, 1});
        p_p2l = model->add_parameters({dim_output, dim_postag});
    }

    if (dim_pretrained_word == 0) {
        std::cerr << "Pretrained word embedding dim should be greater than 0." << std::endl;
        std::cerr << "Pretrained word embedding is inactivated." << std::endl;
        use_pretrained_word = false;
    } else {
        p_t = model->add_lookup_parameters(size_pretrained_word, {dim_pretrained_word, 1});
        for (auto it : pretrained) { p_t->Initialize(it.first, it.second); }
        p_t2l = model->add_parameters({dim_output, dim_pretrained_word});
    }

    p_g2l = model->add_parameters({dim_output, dim_gen});


}


cnn::expr::Expression StaticInputLayer::add_input(cnn::ComputationGraph *hg,
                                                  unsigned wid, unsigned pid, unsigned pre_wid,
                                                  const std::vector<cnn::real> current_gen_feature) {
    cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
    if (use_word && wid > 0) {
        cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
        cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
        expr = cnn::expr::affine_transform({expr, w2l, w});
    }
    if (use_postag && pid > 0) {
        cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
        cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
        expr = cnn::expr::affine_transform({expr, p2l, p});
    }
    if (use_pretrained_word && pre_wid > 0) {
        cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
        cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
        expr = cnn::expr::affine_transform({expr, t2l, t});
    }
    cnn::expr::Expression g2l = cnn::expr::parameter(*hg, p_g2l);
    //cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
    expr = cnn::expr::affine_transform({expr, g2l,
                                        cnn::expr::input(*hg, {static_cast<unsigned int>(current_gen_feature.size())},
                                                         current_gen_feature)});
    return cnn::expr::rectify(expr);
}


DynamicInputLayer::DynamicInputLayer(cnn::Model *model, unsigned dim_gen,
                                     unsigned size_word, unsigned dim_word,
                                     unsigned size_postag, unsigned dim_postag,
                                     unsigned size_pretrained_word, unsigned dim_pretrained_word,
                                     unsigned size_label, unsigned dim_label,
                                     unsigned dim_output,
                                     const std::unordered_map<unsigned, std::vector<float>> &pretrained) :
        StaticInputLayer(model, dim_gen, size_word, dim_word, size_postag, dim_postag,
                         size_pretrained_word, dim_pretrained_word, dim_output, pretrained),
        p_l(nullptr), p_l2l(nullptr), use_label(true) {
    if (dim_label == 0) {
        std::cerr << "Label embedding dim should be greater than 0." << std::endl;
        std::cerr << "Label embedding is inactivated." << std::endl;
        use_label = false;
    } else {
        p_l = model->add_lookup_parameters(size_label, {dim_label, 1});
        p_l2l = model->add_parameters({dim_output, dim_label});
    }
}


cnn::expr::Expression DynamicInputLayer::add_input2(cnn::ComputationGraph *hg,
                                                    unsigned wid, unsigned pid, unsigned pre_wid, unsigned lid) {
    cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
    if (use_word && wid > 0) {
        cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
        cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
        expr = cnn::expr::affine_transform({expr, w2l, w});
    }
    if (use_postag && pid > 0) {
        cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
        cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
        expr = cnn::expr::affine_transform({expr, p2l, p});
    }
    if (use_pretrained_word && pre_wid > 0) {
        cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
        cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
        expr = cnn::expr::affine_transform({expr, t2l, t});
    }
    if (use_label && lid > 0) {
        cnn::expr::Expression l2l = cnn::expr::parameter(*hg, p_l2l);
        cnn::expr::Expression l = cnn::expr::lookup(*hg, p_l, lid);
        expr = cnn::expr::affine_transform({expr, l2l, l});
    }
    return cnn::expr::rectify(expr);
}


cnn::expr::Expression DynamicInputLayer::add_input2(cnn::ComputationGraph *hg,
                                                    unsigned wid, unsigned pid, unsigned pre_wid,
                                                    cnn::expr::Expression &lexpr) {
    cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
    if (use_word && wid > 0) {
        cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
        cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
        expr = cnn::expr::affine_transform({expr, w2l, w});
    }
    if (use_postag && pid > 0) {
        cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
        cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
        expr = cnn::expr::affine_transform({expr, p2l, p});
    }
    if (use_pretrained_word && pre_wid > 0) {
        cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
        cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
        expr = cnn::expr::affine_transform({expr, t2l, t});
    }
    if (use_label) {
        cnn::expr::Expression l2l = cnn::expr::parameter(*hg, p_l2l);
        expr = cnn::expr::affine_transform({expr, l2l, lexpr});
    }
    return cnn::expr::rectify(expr);
}


LSTMLayer::LSTMLayer(cnn::Model *model,
                     unsigned n_layers,
                     unsigned dim_input,
                     unsigned dim_hidden,
                     bool rev) : n_items(0),
                                 lstm(n_layers, dim_input, dim_hidden, model),
                                 p_guard(model->add_parameters({dim_input, 1})),
                                 reversed(rev) {
}


RNNLanguageModel::RNNLanguageModel(cnn::Model *model_lm,
                     unsigned VOCAB_SIZE,
                     unsigned LAYERS,
                     unsigned INPUT_DIM,
                     unsigned HIDDEN_DIM):
        builders(LAYERS, INPUT_DIM,  HIDDEN_DIM, model_lm),
        p_c(model_lm->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
        p_R(model_lm->add_parameters({VOCAB_SIZE, HIDDEN_DIM})),
        p_bias(model_lm->add_parameters({VOCAB_SIZE}))
      {

      }

void RNNLanguageModel::BuildLMGraphs(const std::vector<std::vector<unsigned >>& sents,
                       unsigned id,
                       unsigned & chars,
                       unsigned bsize,
                       cnn::ComputationGraph* hg){
    const unsigned slen = sents[id].size();



//    Tensor current_gen_feature  =  p_R->values;

    builders.new_graph(*hg);  // reset RNN builder for new graph
    builders.start_new_sequence();
    cnn::expr::Expression i_R = cnn::expr::parameter(*hg, p_R); // hidden -> word rep parameter
    cnn::expr::Expression i_bias = cnn::expr::parameter(*hg, p_bias);  // word bias
    std::vector<cnn::expr::Expression> errs;
    std::vector<unsigned> last_arr(bsize, sents[0][0]), next_arr(bsize);

//    cnn::Tensor temp = p_R->values;
//    std::vector<float>  w = temp::;
//    std::vector<unsigned> temp;
//    temp.push_back(1);
//    cnn::expr::Expression exp1 = cnn::expr::select_rows(i_R,temp);
//    std::vector<float> ww = cnn::as_vector(hg->get_value(exp1));
//
//    cnn::expr::Expression exp2;


   // std::vector<cnn::Parameters*>  p_W3 = cnn::expr::;

    for (unsigned t = 1; t < slen; ++t) {
        for (unsigned i = 0; i < bsize; ++i) {
            next_arr[i] = sents[id+i][t];
            if(next_arr[i] != *sents[id].rbegin()) chars++; // add non-EOS
        }
        // y_t = RNN(x_t)
        cnn::expr::Expression i_x_t = cnn::expr::lookup(*hg, p_c, last_arr);
        cnn::expr::Expression i_y_t = builders.add_input(i_x_t);
        cnn::expr::Expression i_r_t = i_bias + i_R * i_y_t;
        cnn::expr::Expression i_err = cnn::expr::pickneglogsoftmax(i_r_t, next_arr);
        errs.push_back(i_err);
        last_arr = next_arr;
    }
    cnn::expr::Expression i_nerr = cnn::expr::sum_batches(sum(errs));
}


cnn::expr::Expression RNNLanguageModel::new_graph(cnn::ComputationGraph* hg, unsigned sos){
  //  const unsigned slen = sents[id].size();




    builders.new_graph(*hg);  // reset RNN builder for new graph
    builders.start_new_sequence();
    cnn::expr::Expression i_x_t = cnn::expr::const_lookup(*hg, p_c, sos);
    cnn::expr::Expression i_y_t = builders.add_input(i_x_t);
    return i_y_t;
}

cnn::expr::Expression RNNLanguageModel::add_input(cnn::ComputationGraph* hg, unsigned word){
    //  const unsigned slen = sents[id].size();
    cnn::expr::Expression i_x_t = cnn::expr::const_lookup(*hg, p_c, word);
    cnn::expr::Expression i_y_t = builders.add_input(i_x_t);
    return i_y_t;

}



cnn::expr::Expression RNNLanguageModel::get_iR(cnn::ComputationGraph *hg){
    cnn::expr::Expression i_R = cnn::expr::parameter(*hg, p_R); // hidden -> word rep parameter
    return i_R;
}







void RNNLanguageModel::disable_dropout() {
    builders.disable_dropout();
}
void RNNLanguageModel::set_dropout(float &rate) {
    builders.set_dropout(rate);
}

//RNNLanguageModel::RNNLanguageModel(cnn::Model *model,
//                     unsigned VOCAB_SIZE,
//                     unsigned LAYERS,
//                     unsigned INPUT_DIM,
//                     unsigned HIDDEN_DIM) {
//    builders(LAYERS, INPUT_DIM,  HIDDEN_DIM, model);
//    p_c = model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM});
//    p_R = model->add_parameters({VOCAB_SIZE, HIDDEN_DIM});
//    p_bias = model->add_parameters({VOCAB_SIZE});
//}

//p_l = model->add_lookup_parameters(size_label, {dim_label, 1});


void LSTMLayer::new_graph(cnn::ComputationGraph *hg) {
    lstm.new_graph(*hg);
}


void LSTMLayer::add_inputs(cnn::ComputationGraph *hg,
                           const std::vector<cnn::expr::Expression> &inputs,const std::vector<cnn::expr::Expression> &encdec) {
    n_items = inputs.size();
    if(encdec.size() > 0){
        lstm.start_new_sequence(encdec);
    }
    else{
        lstm.start_new_sequence();
    }


    lstm.add_input(cnn::expr::parameter(*hg, p_guard));
    if (reversed) {
        for (int i = n_items - 1; i >= 0; --i) {
            lstm.add_input(inputs[i]);
        }
    } else {
        for (unsigned i = 0; i < n_items; ++i) {
            lstm.add_input(inputs[i]);
        }
    }
}


cnn::expr::Expression LSTMLayer::get_output(cnn::ComputationGraph *hg, int index) {
    if (reversed) {
        return lstm.get_h(cnn::RNNPointer(n_items - index)).back();
    }
    return lstm.get_h(cnn::RNNPointer(index + 1)).back();
}


void LSTMLayer::get_outputs(cnn::ComputationGraph *hg,
                            std::vector<cnn::expr::Expression> &outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) {
        outputs[i] = get_output(hg, i);
    }
}


void LSTMLayer::set_dropout(float &rate) {
    lstm.set_dropout(rate);
}

void LSTMLayer::disable_dropout() {
    lstm.disable_dropout();
}

BidirectionalLSTMLayer::BidirectionalLSTMLayer(cnn::Model *model,
                                               unsigned n_lstm_layers,
                                               unsigned dim_lstm_input,
                                               unsigned dim_hidden) :
        n_items(0),
        fw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
        bw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
        p_fw_guard(model->add_parameters({dim_lstm_input, 1})),
        p_bw_guard(model->add_parameters({dim_lstm_input, 1})) {

}


void BidirectionalLSTMLayer::new_graph(cnn::ComputationGraph *hg) {
    fw_lstm.new_graph(*hg);
    bw_lstm.new_graph(*hg);
}



void BidirectionalLSTMLayer::add_inputs(cnn::ComputationGraph *hg,
                                        const std::vector<cnn::expr::Expression> &inputs) {
    n_items = inputs.size();
    fw_lstm.start_new_sequence();
    bw_lstm.start_new_sequence();

    fw_lstm.add_input(cnn::expr::parameter(*hg, p_fw_guard));
    for (unsigned i = 0; i < n_items; ++i) {
        fw_lstm.add_input(inputs[i]);
        bw_lstm.add_input(inputs[n_items - i - 1]);
    }
    bw_lstm.add_input(cnn::expr::parameter(*hg, p_bw_guard));
}


BidirectionalLSTMLayer::Output BidirectionalLSTMLayer::get_output(cnn::ComputationGraph *hg,
                                                                  int index) {
    return std::make_pair(
            fw_lstm.get_h(cnn::RNNPointer(index + 1)).back(),
            bw_lstm.get_h(cnn::RNNPointer(n_items - index - 1)).back());
}


void BidirectionalLSTMLayer::get_outputs(cnn::ComputationGraph *hg,
                                         std::vector<BidirectionalLSTMLayer::Output> &outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) {
        outputs[i] = get_output(hg, i);
    }
}


void BidirectionalLSTMLayer::set_dropout(float &rate) {
    fw_lstm.set_dropout(rate);
    bw_lstm.set_dropout(rate);
}


void BidirectionalLSTMLayer::disable_dropout() {
    fw_lstm.disable_dropout();
    bw_lstm.disable_dropout();
}


SoftmaxLayer::SoftmaxLayer(cnn::Model *model,
                           unsigned dim_input,
                           unsigned dim_output)
        : p_B(model->add_parameters({dim_output, 1})),
          p_W(model->add_parameters({dim_output, dim_input})) {
}


cnn::expr::Expression SoftmaxLayer::get_output(cnn::ComputationGraph *hg,
                                               const cnn::expr::Expression &expr) {
    return cnn::expr::log_softmax(cnn::expr::affine_transform({
                                                                      cnn::expr::parameter(*hg, p_B),
                                                                      cnn::expr::parameter(*hg, p_W), expr}));
}


DenseLayer::DenseLayer(cnn::Model *model,
                       unsigned dim_input,
                       unsigned dim_output) :
        p_W(model->add_parameters({dim_output, dim_input})),
        p_B(model->add_parameters({dim_output, 1})) {

}


cnn::expr::Expression DenseLayer::get_output(cnn::ComputationGraph *hg,
                                             const cnn::expr::Expression &expr) {
    return cnn::expr::affine_transform({
                                               cnn::expr::parameter(*hg, p_B),
                                               cnn::expr::parameter(*hg, p_W),
                                               expr});
}


Merge2Layer::Merge2Layer(cnn::Model *model,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
                                                p_W1(model->add_parameters({dim_output, dim_input1})),
                                                p_W2(model->add_parameters({dim_output, dim_input2})) {
}


cnn::expr::Expression Merge2Layer::get_output(cnn::ComputationGraph *hg,
                                              const cnn::expr::Expression &expr1,
                                              const cnn::expr::Expression &expr2) {
    cnn::expr::Expression i = cnn::expr::affine_transform({
                                                                  cnn::expr::parameter(*hg, p_B),
                                                                  cnn::expr::parameter(*hg, p_W1), expr1,
                                                                  cnn::expr::parameter(*hg, p_W2), expr2
                                                          });
    return i;
}

//struct Merge_enc_Layer {
//    cnn::Parameters *p_ie2h;
//    cnn::Parameters *p_bie;
//    cnn::Parameters *p_h2oe;
//    cnn::Parameters *p_boe;
//
//
//    Merge_enc_Layer(cnn::Model* model,
//                    unsigned HIDDEN_DIM,
//                    unsigned LAYERS);
//
//    cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
//                                     const cnn::expr::Expression& expr1);
//};




Merge_enc_Layer::Merge_enc_Layer(cnn::Model *model,
                         unsigned HIDDEN_DIM,
                         unsigned LAYERS) : p_ie2h(model->add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * 2)})),
                                            p_bie(model->add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)})),
                                            p_h2oe(model->add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)})),
                                            p_boe(model->add_parameters({unsigned(HIDDEN_DIM * LAYERS)})),
                                            layers(LAYERS),
                                            hidden_dim(HIDDEN_DIM) {
}


std::vector<cnn::expr::Expression> Merge_enc_Layer::get_output(cnn::ComputationGraph *hg,
                                                  const cnn::expr::Expression &expr1) {
    cnn::expr::Expression i_ie2h = parameter(*hg, p_ie2h);
    cnn::expr::Expression i_bie = parameter(*hg, p_bie);
    cnn::expr::Expression i_t = i_bie + i_ie2h * expr1;
    hg->incremental_forward();
    cnn::expr::Expression i_h = rectify(i_t);
    cnn::expr::Expression i_h2oe = parameter(*hg,p_h2oe);
    cnn::expr::Expression i_boe = parameter(*hg,p_boe);
    cnn::expr::Expression i_nc = i_boe + i_h2oe * i_h;

    std::vector<Expression> oein1, oein2, oein;
    for (unsigned i = 0; i < layers; ++i) {
        oein1.push_back(pickrange(i_nc, i * hidden_dim, (i + 1) * hidden_dim));
        oein2.push_back(tanh(oein1[i]));
    }
    for (unsigned i = 0; i < layers; ++i) oein.push_back(oein1[i]);
    for (unsigned i = 0; i < layers; ++i) oein.push_back(oein2[i]);
    return oein;
}






//Merge3Layer::Merge3Layer(cnn::Model *model,
//                         unsigned dim_input1,
//                         unsigned dim_input2,
//                         unsigned dim_input3,
//                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
//                                                p_W1(model->add_parameters({dim_output, dim_input1})),
//                                                p_W2(model->add_parameters({dim_output, dim_input2})),
//                                                p_W3(model->add_parameters({dim_output, dim_input3})) {
//}
Merge3Layer::Merge3Layer(cnn::Model *model,unsigned word_dim,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_output):
        p_B(nullptr),p_W1(nullptr),p_W2(nullptr) {
       p_B = model->add_parameters({dim_output, 1});
       p_W1 = model->add_parameters({dim_output, dim_input1});
       p_W2 = model->add_parameters({dim_output, dim_input2});
       p_W3 = model->add_lookup_parameters(word_dim, {dim_output, dim_input3});
//       p_W3.resize(word_dim);
//       for (unsigned i = 0; i < word_dim; i++) {
//           p_W3[i] = model->add_parameters({dim_output, dim_input3});
//      }
          //  p_W3(model->add_parameters({dim_output, dim_input3});
}


cnn::expr::Expression Merge3Layer::get_output(cnn::ComputationGraph *hg,
                                              unsigned wid,
                                              const cnn::expr::Expression &expr1,
                                              const cnn::expr::Expression &expr2,
                                              const cnn::expr::Expression &expr3) {
    return cnn::expr::affine_transform({
                                               cnn::expr::parameter(*hg, p_B),
                                               cnn::expr::parameter(*hg, p_W1), expr1,
                                               cnn::expr::parameter(*hg, p_W2), expr2,
                                               //cnn::expr::parameter(*hg, p_W3[wid]), expr3,
                                               cnn::expr::lookup(*hg, p_W3, wid), expr3,
                                       });
}

void Merge3Layer::W3_initial(cnn::ComputationGraph *hg, const cnn::expr::Expression& expr1) {
    for (unsigned i = 0; i < p_W3->size(); i++) {
        std::vector<unsigned> temp;
        temp.push_back(i);
        cnn::expr::Expression exp2 = cnn::expr::select_rows(expr1,temp);
        std::vector<float> ww = cnn::as_vector(hg->get_value(exp2));
        p_W3->Initialize(i,ww);
    }
}
//Merge3Layer_new::Merge3Layer_new(cnn::Model *model,unsigned word_dim,
//                         unsigned dim_input1,
//                         unsigned dim_input2,
//                         unsigned dim_input3,
//                         unsigned dim_output):
//        p_B(nullptr),p_W1(nullptr),p_W2(nullptr),p_W3(nullptr) {
//       p_B = model->add_parameters({dim_output, 1});
//       p_W1 = model->add_parameters({dim_output, dim_input1});
//       p_W2 = model->add_parameters({dim_output, dim_input2});
//       p_W3 = model->add_lookup_parameters(word_dim, {dim_output, dim_input3});
////    p_W3.resize(word_dim);
////    for (unsigned i = 0; i < word_dim; i++) {
////        p_W3[i] = model->add_parameters({dim_output, dim_input3});
////    }
//          //  p_W3(model->add_parameters({dim_output, dim_input3});
//}
//
//
//cnn::expr::Expression Merge3Layer_new::get_output(cnn::ComputationGraph *hg,unsigned wid,
//                                              const cnn::expr::Expression &expr1,
//                                              const cnn::expr::Expression &expr2,
//                                              const cnn::expr::Expression &expr3) {
//    //cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
//    return cnn::expr::affine_transform({
//                                               cnn::expr::parameter(*hg, p_B),
//                                               cnn::expr::parameter(*hg, p_W1), expr1,
//                                               cnn::expr::parameter(*hg, p_W2), expr2,
//                                               cnn::expr::lookup(*hg, p_W3, wid), expr3
//                                       });
//}





//Merge4Layer::Merge4Layer(cnn::Model *model,
//                         unsigned dim_input1,
//                         unsigned dim_input2,
//                         unsigned dim_input3,
//                         unsigned dim_input4,
//                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
//                                                p_W1(model->add_parameters({dim_output, dim_input1})),
//                                                p_W2(model->add_parameters({dim_output, dim_input2})),
//                                                p_W3(model->add_parameters({dim_output, dim_input3})),
//                                                p_W4(model->add_parameters({dim_output, dim_input4})) {
//}
//
//
//cnn::expr::Expression Merge4Layer::get_output(cnn::ComputationGraph *hg,
//                                              const cnn::expr::Expression &expr1,
//                                              const cnn::expr::Expression &expr2,
//                                              const cnn::expr::Expression &expr3,
//                                              const cnn::expr::Expression &expr4) {
//    return cnn::expr::affine_transform({
//                                               cnn::expr::parameter(*hg, p_B),
//                                               cnn::expr::parameter(*hg, p_W1), expr1,
//                                               cnn::expr::parameter(*hg, p_W2), expr2,
//                                               cnn::expr::parameter(*hg, p_W3), expr3,
//                                               cnn::expr::parameter(*hg, p_W4), expr4
//                                       });
//}





//Merge4Layer::Merge4Layer(cnn::Model *model,
//                         unsigned word_dim,
//                         unsigned dim_input1,
//                         unsigned dim_input2,
//                         unsigned dim_input3,
//                         unsigned dim_input4,
//                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
//                                                p_W1(model->add_parameters({dim_output, dim_input1})),
//                                                p_W2(model->add_parameters({dim_output, dim_input2})),
//                                                p_W3(model->add_lookup_parameters(word_dim, {dim_output, dim_input3})),
//                                                p_W4(model->add_parameters({dim_output, dim_input4})) {
//}
//
//
//cnn::expr::Expression Merge4Layer::get_output(cnn::ComputationGraph *hg,
//                                              unsigned wid,
//                                              const cnn::expr::Expression &expr1,
//                                              const cnn::expr::Expression &expr2,
//                                              const cnn::expr::Expression &expr3,
//                                              const cnn::expr::Expression &expr4) {
//    return cnn::expr::affine_transform({
//                                               cnn::expr::parameter(*hg, p_B),
//                                               cnn::expr::parameter(*hg, p_W1), expr1,
//                                               cnn::expr::parameter(*hg, p_W2), expr2,
//                                               cnn::expr::lookup(*hg, p_W3, wid), expr3,
//                                               cnn::expr::parameter(*hg, p_W4), expr4
//                                       });
//}


Merge4Layer::Merge4Layer(cnn::Model *model, unsigned word_dim,
                                 unsigned dim_input1,
                                 unsigned dim_input2,
                                 unsigned dim_input3,
                                 unsigned dim_input4,
                                 unsigned dim_output) :
        p_B(nullptr),p_W1(nullptr),p_W2(nullptr),p_W4(nullptr)  {

    p_B = model->add_parameters({dim_output, 1});
    p_W1 = model->add_parameters({dim_output, dim_input1});
    p_W2 = model->add_parameters({dim_output, dim_input2});
    p_W3 = model->add_lookup_parameters(word_dim, {dim_output, dim_input3});
    //p_W3 = model->add_parameters({ dim_output, dim_input3 });
    p_W4 = model->add_parameters({dim_output, dim_input4});

   // p_W3.resize(word_dim);
//    for (unsigned i = 0; i < word_dim; i++) {
//        p_W3[i] = model->add_parameters({dim_output, dim_input3});
//    }

//    for (unsigned i = 0; i < word_dim; i++) {
//        p_W3.push_back(model->add_parameters({dim_output, dim_input3}));
//    }


}

void Merge4Layer::W3_initial(cnn::ComputationGraph *hg, const cnn::expr::Expression& expr1) {
    for (unsigned i = 0; i < p_W3->size(); i++) {
        std::vector<unsigned> temp;
        temp.push_back(i);
        cnn::expr::Expression exp2 = cnn::expr::select_rows(expr1,temp);
        std::vector<float> ww = cnn::as_vector(hg->get_value(exp2));
        p_W3->Initialize(i,ww);
    }
}


cnn::expr::Expression Merge4Layer::get_output(cnn::ComputationGraph *hg,
                                                  unsigned wid,
                                                  const cnn::expr::Expression &expr1,
                                                  const cnn::expr::Expression &expr2,
                                                  const cnn::expr::Expression &expr3,
                                                  const cnn::expr::Expression &expr4) {

    return cnn::expr::affine_transform({
                                               cnn::expr::parameter(*hg, p_B),
                                               cnn::expr::parameter(*hg, p_W1), expr1,
                                               cnn::expr::parameter(*hg, p_W2), expr2,
                                              // cnn::expr::parameter(*hg, p_W3[wid]), expr3,
                                               cnn::expr::lookup(*hg, p_W3, wid), expr3,
                                               cnn::expr::parameter(*hg, p_W4), expr4
                                       });
}


Merge5Layer::Merge5Layer(cnn::Model *model,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
                                                p_W1(model->add_parameters({dim_output, dim_input1})),
                                                p_W2(model->add_parameters({dim_output, dim_input2})),
                                                p_W3(model->add_parameters({dim_output, dim_input3})),
                                                p_W4(model->add_parameters({dim_output, dim_input4})),
                                                p_W5(model->add_parameters({dim_output, dim_input5})) {
}


cnn::expr::Expression Merge5Layer::get_output(cnn::ComputationGraph *hg,
                                              const cnn::expr::Expression &expr1,
                                              const cnn::expr::Expression &expr2,
                                              const cnn::expr::Expression &expr3,
                                              const cnn::expr::Expression &expr4,
                                              const cnn::expr::Expression &expr5) {
    return cnn::expr::affine_transform({
                                               cnn::expr::parameter(*hg, p_B),
                                               cnn::expr::parameter(*hg, p_W1), expr1,
                                               cnn::expr::parameter(*hg, p_W2), expr2,
                                               cnn::expr::parameter(*hg, p_W3), expr3,
                                               cnn::expr::parameter(*hg, p_W4), expr4,
                                               cnn::expr::parameter(*hg, p_W5), expr5
                                       });
}


Merge6Layer::Merge6Layer(cnn::Model *model,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_input6,
                         unsigned dim_output) : p_B(model->add_parameters({dim_output, 1})),
                                                p_W1(model->add_parameters({dim_output, dim_input1})),
                                                p_W2(model->add_parameters({dim_output, dim_input2})),
                                                p_W3(model->add_parameters({dim_output, dim_input3})),
                                                p_W4(model->add_parameters({dim_output, dim_input4})),
                                                p_W5(model->add_parameters({dim_output, dim_input5})),
                                                p_W6(model->add_parameters({dim_output, dim_input6})) {
}


cnn::expr::Expression Merge6Layer::get_output(cnn::ComputationGraph *hg,
                                              const cnn::expr::Expression &expr1,
                                              const cnn::expr::Expression &expr2,
                                              const cnn::expr::Expression &expr3,
                                              const cnn::expr::Expression &expr4,
                                              const cnn::expr::Expression &expr5,
                                              const cnn::expr::Expression &expr6) {
    return cnn::expr::affine_transform({
                                               cnn::expr::parameter(*hg, p_B),
                                               cnn::expr::parameter(*hg, p_W1), expr1,
                                               cnn::expr::parameter(*hg, p_W2), expr2,
                                               cnn::expr::parameter(*hg, p_W3), expr3,
                                               cnn::expr::parameter(*hg, p_W4), expr4,
                                               cnn::expr::parameter(*hg, p_W5), expr5,
                                               cnn::expr::parameter(*hg, p_W6), expr6
                                       });
}


SegUniEmbedding::SegUniEmbedding(cnn::Model &m, unsigned n_layers,
                                 unsigned lstm_input_dim, unsigned seg_dim)
        :
        p_h0(m.add_parameters({lstm_input_dim})),
        builder(n_layers, lstm_input_dim, seg_dim, &m) {
}

void SegUniEmbedding::construct_chart(cnn::ComputationGraph &cg,
                                      const std::vector<cnn::expr::Expression> &c,
                                      int max_seg_len) {
    len = c.size();
    h.clear(); // The first dimension for h is the starting point, the second is length.
    h.resize(len);
    cnn::expr::Expression h0 = cnn::expr::parameter(cg, p_h0);
    builder.new_graph(cg);
    for (unsigned i = 0; i < len; ++i) {
        unsigned max_j = i + len;
        if (max_seg_len) { max_j = i + max_seg_len; }
        if (max_j > len) { max_j = len; }
        unsigned seg_len = max_j - i;
        auto &hi = h[i];
        hi.resize(seg_len + 1);

        builder.start_new_sequence();
        hi[0] = builder.add_input(h0);
        // Put one span in h[i][j]
        for (unsigned k = 0; k < seg_len; ++k) {
            hi[k + 1] = builder.add_input(c[i + k]);
        }
    }
}

const cnn::expr::Expression &SegUniEmbedding::operator()(unsigned i, unsigned j) const {
    BOOST_ASSERT(j <= len);
    BOOST_ASSERT(j >= i);
    return h[i][j - i];
}

void SegUniEmbedding::set_dropout(float &rate) {
    builder.set_dropout(rate);
}

void SegUniEmbedding::disable_dropout() {
    builder.disable_dropout();
}

SegBiEmbedding::SegBiEmbedding(cnn::Model &m, unsigned n_layers,
                               unsigned lstm_input_dim, unsigned seg_dim)
        :
        fwd(m, n_layers, lstm_input_dim, seg_dim),
        bwd(m, n_layers, lstm_input_dim, seg_dim) {
}

void SegBiEmbedding::construct_chart(cnn::ComputationGraph &cg,
                                     const std::vector<cnn::expr::Expression> &c,
                                     int max_seg_len) {
    len = c.size();
    fwd.construct_chart(cg, c, max_seg_len);
    std::vector<cnn::expr::Expression> rc(len);
    for (unsigned i = 0; i < len; ++i) { rc[i] = c[len - i - 1]; }
    bwd.construct_chart(cg, rc, max_seg_len);
    h.clear();
    h.resize(len);
    for (unsigned i = 0; i < len; ++i) {
        unsigned max_j = i + len;
        if (max_seg_len) { max_j = i + max_seg_len; }
        if (max_j > len) { max_j = len; }
        auto &hi = h[i];
        unsigned seg_len = max_j - i;
        hi.resize(seg_len + 1);
        const cnn::expr::Expression &fe = fwd(i, i);
        const cnn::expr::Expression &be = bwd(len - 1, len - 1);
        hi[0] = std::make_pair(fe, be);
        for (unsigned k = 0; k < seg_len; ++k) {
            unsigned j = i + k;
            const cnn::expr::Expression &fe = fwd(i, j + 1);
            const cnn::expr::Expression &be = bwd(len - 1 - j, len - 1 - i);
            hi[k + 1] = std::make_pair(fe, be);
        }
    }
}

const SegBiEmbedding::ExpressionPair &SegBiEmbedding::operator()(unsigned i, unsigned j) const {
    BOOST_ASSERT(j <= len);
    BOOST_ASSERT(j >= i);
    return h[i][j - i];
}

void SegBiEmbedding::set_dropout(float &rate) {
    fwd.set_dropout(rate);
    bwd.set_dropout(rate);
}

void SegBiEmbedding::disable_dropout() {
    fwd.disable_dropout();
    bwd.disable_dropout();
}
