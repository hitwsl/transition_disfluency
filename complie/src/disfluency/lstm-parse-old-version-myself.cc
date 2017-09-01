#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "logging.h"
#include "cnn/rnn.h"
#include "stack-lstm-beam-search/c2.h"
#include "training_utils.h"

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned WORD_DIM = 100;
unsigned CHAR_DIM = 30;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;
unsigned GEN_DIM = 78;
unsigned BEAM_SIZE = 8;


//unsigned LSTM_CHAR_OUTPUT_DIM = 100; //Miguel
bool USE_SPELLING = false;

float DROPOUT = 0.0f;

bool USE_POS = false;

constexpr const char *ROOT_SYMBOL = "ROOT";
unsigned kROOT_SYMBOL = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;

unsigned CHAR_SIZE = 255; //size of ascii chars... Miguel

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char **argv, po::variables_map *conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
            ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
            ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
            ("eta", po::value<float>()->default_value(0.1), "the eta value.")
            ("eta_update", po::value<float>()->default_value(1.0), "the eta value.")
            ("dev_data,d", po::value<string>(), "Development corpus")
            ("test_data,p", po::value<string>(), "Test corpus")
            //("beam_size", po::value<unsigned>()->default_value(4), "beam size")
            ("dev_result", po::value<string>()->default_value("dev_result"), "Test corpus")
            ("test_result", po::value<string>()->default_value("test_result"), "Test corpus")
            ("dropout,D", po::value<float>(), "Dropout rate")
            ("unk_strategy,o", po::value<unsigned>()->default_value(1),
             "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
            ("gen_fea_dim", po::value<unsigned>()->default_value(78), "generalization_feature dim")
            ("unk_prob,u", po::value<double>()->default_value(0.2),
             "Probably with which to replace singletons with UNK in training data")
            ("model,m", po::value<string>(), "Load saved model from this file")
            ("use_pos_tags,P", "make POS tags visible to parser")
            ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
            ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
            ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
            //("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
            ("word_dim", po::value<unsigned>()->default_value(100), "input embedding size")
            ("char_dim", po::value<unsigned>()->default_value(32), "input embedding size")
            ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
            ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
            ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
            ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
            ("maxiter", po::value<unsigned>()->default_value(10), "Max number of iterations.")
            ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
            ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
            ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
            ("train,t", "Should training be run?")
            ("words,w", po::value<string>(), "Pretrained word embeddings")
            ("use_spelling,S", "Use spelling model") //Miguel. Spelling model
            ("help,h", "Help");
    po::options_description dcmdline_options;
    dcmdline_options.add(opts);
    po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
    if (conf->count("help")) {
        cerr << dcmdline_options << endl;
        exit(1);
    }
    if (conf->count("training_data") == 0) {
        cerr <<
        "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
        exit(1);
    }
}

struct ParserBuilder {

    LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
    LSTMBuilder output_lstm; // (layers, input, hidden, trainer)
    LSTMBuilder buffer_lstm;
    LSTMBuilder action_lstm;


    LSTMBuilder ent_lstm_fwd;
    LSTMBuilder ent_lstm_rev;


    LookupParameters *p_w; // word embeddings
    LookupParameters *p_t; // pretrained word embeddings (not updated)
    LookupParameters *p_a; // input action embeddings
    LookupParameters *p_r; // relation embeddings
    LookupParameters *p_p; // pos tag embeddings
    Parameters *p_pbias; // parser state bias
    Parameters *p_A; // action lstm to parser state
    Parameters *p_B; // buffer lstm to parser state
    Parameters *p_O; // output lstm to parser state

    Parameters *p_S; // stack lstm to parser state
    Parameters *p_H; // head matrix for composition function
    Parameters *p_D; // dependency matrix for composition function
    Parameters *p_R; // relation matrix for composition function
    Parameters *p_w2l; // word to LSTM input
    Parameters *s_w2l; // word to LSTM input
    Parameters *p_g2l; // parameter for gen feature
    Parameters *p_p2l; // POS to LSTM input
    Parameters *p_t2l; // pretrained word embeddings to LSTM input
    Parameters *p_ib; // LSTM input bias
    Parameters *p_cbias; // composition function bias
    Parameters *p_p2a;   // parser state to action
    Parameters *p_action_start;  // action bias
    Parameters *p_abias;  // action bias
    Parameters *p_buffer_guard;  // end of buffer
    Parameters *p_stack_guard;  // end of stack
    Parameters *p_output_guard;  // end of output buffer

    Parameters *p_start_of_word;
    //Miguel -->dummy <s> symbol
    Parameters *p_end_of_word; //Miguel --> dummy </s> symbol
    LookupParameters *char_emb; //Miguel-> mapping of characters to vectors


    LSTMBuilder fw_char_lstm; // Miguel
    LSTMBuilder bw_char_lstm; //Miguel

    Parameters *p_cW;


    explicit ParserBuilder(Model *model, const unordered_map<unsigned, vector<float>> &pretrained) :
            stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
            buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
            output_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
            action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
            ent_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model),
            ent_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model),
            p_w(model->add_lookup_parameters(VOCAB_SIZE, {WORD_DIM})),
            p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
            p_r(model->add_lookup_parameters(ACTION_SIZE, {REL_DIM})),
            p_pbias(model->add_parameters({HIDDEN_DIM})),
            p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
            p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
            p_O(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
            p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
            p_H(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
            p_D(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
            p_R(model->add_parameters({LSTM_INPUT_DIM, REL_DIM})),
            p_w2l(model->add_parameters({LSTM_INPUT_DIM, WORD_DIM})),
            s_w2l(model->add_parameters({LSTM_INPUT_DIM, WORD_DIM})),
            p_ib(model->add_parameters({LSTM_INPUT_DIM})),
            p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
            p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
            p_action_start(model->add_parameters({ACTION_DIM})),
            p_abias(model->add_parameters({ACTION_SIZE})),

            p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
            p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),
            p_output_guard(model->add_parameters({LSTM_INPUT_DIM})),

            p_start_of_word(model->add_parameters({CHAR_DIM})), //Miguel
            p_end_of_word(model->add_parameters({CHAR_DIM})), //Miguel

            char_emb(model->add_lookup_parameters(CHAR_SIZE, {CHAR_DIM})),//Miguel

            p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})), //ner.

//      fw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM, model), //Miguel
//      bw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM,  model), //Miguel

            fw_char_lstm(LAYERS, CHAR_DIM, WORD_DIM / 2, model), //Miguel
            bw_char_lstm(LAYERS, CHAR_DIM, WORD_DIM / 2, model) /*Miguel*/ {
        if (USE_POS) {
            p_p = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
            p_p2l = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
        }
        p_g2l = model->add_parameters({LSTM_INPUT_DIM, GEN_DIM});
        if (pretrained.size() > 0) {
            p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
            for (auto it : pretrained)
                p_t->Initialize(it.first, it.second);
            p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
        } else {
            p_t = nullptr;
            p_t2l = nullptr;
        }
    }

    static bool IsActionForbidden(const string &a, unsigned bsize, unsigned ssize, vector<int> stacki) {

        bool is_shift = (a[0] == 'S');  //MIGUEL
        bool is_reduce = (a[0] == 'R');
        bool is_output = (a[0] == 'O');
//  std::cout<<"is red:"<<is_reduce<<"\n";
        if (is_shift && bsize == 1) return true;
        if (is_reduce && ssize == 1) return true;
        if (is_output && bsize == 1) return true;

        return false;
    }

    map<int, string> compute_ents(unsigned sent_len, const vector<unsigned> &actions,
                                  const vector<string> &setOfActions, map<int, string> *pr = nullptr) {
        map<int, string> r;
        map<int, string> &rels = (pr ? *pr : r);
        for (unsigned i = 0; i < sent_len; i++) { rels[i] = "ERROR"; }
        vector<int> bufferi(sent_len + 1, 0);
        for (unsigned i = 0; i < sent_len; ++i)
            bufferi[sent_len - i] = i;
        bufferi[0] = -999;

        for (auto action: actions) { // loop over transitions for sentence
            const string &actionString = setOfActions[action];
            // std::cout<<"int"<<action<<"-"<<"actionString"<<actionString<<"\n";
            const char ac = actionString[0];
            if (ac == 'S') {  // SHIFT
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                rels[bufferi.back()] = "REDUCE";
                bufferi.pop_back();
            }
            else if (ac == 'O') {
                assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                rels[bufferi.back()] = "0UT";
                bufferi.pop_back();
            }
        }
        assert(bufferi.size() == 1);
        return rels;
    }

    struct ParserState {
        LSTMBuilder stack_lstm;
        LSTMBuilder buffer_lstm;
        LSTMBuilder output_lstm;
        LSTMBuilder action_lstm;
        vector<Expression> buffer;
        vector<int> bufferi;
        vector<Expression> stack;
        vector<int> stacki;
        vector<Expression> output;
        vector<int> outputi;
        vector<unsigned> results;  // sequence of predicted actions
        bool complete;

        Expression score_exp;
        double score;
        bool is_gold;
    };


    struct ParserStateCompare {
        bool operator()(const ParserState &a, const ParserState &b) const {
            return a.score > b.score;
        }
    };

    static void prune(vector<ParserState> &pq, unsigned k) {
        if (pq.size() == 1) return;
        if (k > pq.size()) k = pq.size();
        partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
        pq.resize(k);
        reverse(pq.begin(), pq.end());
        //cerr << "PRUNE\n";
        //for (unsigned i = 0; i < pq.size(); ++i) {
        //  cerr << pq[i].score << endl;
        //}
    }

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
    inline unsigned int UTF8Len(unsigned char x) {
        if (x < 0x80) return 1;
        else if ((x >> 5) == 0x06) return 2;
        else if ((x >> 4) == 0x0e) return 3;
        else if ((x >> 3) == 0x1e) return 4;
        else if ((x >> 2) == 0x3e) return 5;
        else if ((x >> 1) == 0x7e) return 6;
        else return 0;
    }


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
    vector<unsigned> log_prob_parser_beam_train(ComputationGraph *hg,
                                                const std::vector<std::vector<cnn::real>> current_gen_feature,
                                                const vector<string> &str_sent,  // raw sentence
                                                const vector<unsigned> &raw_sent,  // raw sentence
                                                const vector<unsigned> &sent,  // sent with oovs replaced
                                                const vector<unsigned> &sentPos,
                                                const vector<unsigned> &correct_actions,
                                                const vector<string> &setOfActions,
                                                const map<unsigned, std::string> &intToWords,
                                                bool is_evaluation,
                                                double *right) {
        //for (unsigned i = 0; i < sent.size(); ++i) cerr << ' ' << intToWords.find(sent[i])->second;
        //cerr << endl;
        vector<unsigned> results;
        const bool build_training_graph = correct_actions.size() > 0;
        //std::cout<<"****************"<<"\n";
        bool apply_dropout = (DROPOUT && !is_evaluation);
        ParserState init;

        stack_lstm.new_graph(*hg);
        buffer_lstm.new_graph(*hg);
        output_lstm.new_graph(*hg);
        action_lstm.new_graph(*hg);

        ent_lstm_fwd.new_graph(*hg);
        ent_lstm_rev.new_graph(*hg);


        if (apply_dropout) {
            stack_lstm.set_dropout(DROPOUT);
            action_lstm.set_dropout(DROPOUT);
            buffer_lstm.set_dropout(DROPOUT);
            ent_lstm_fwd.set_dropout(DROPOUT);
            ent_lstm_rev.set_dropout(DROPOUT);
        } else {
            stack_lstm.disable_dropout();
            action_lstm.disable_dropout();
            buffer_lstm.disable_dropout();
            ent_lstm_fwd.disable_dropout();
            ent_lstm_rev.disable_dropout();
        }

        stack_lstm.start_new_sequence();
        buffer_lstm.start_new_sequence();
        output_lstm.start_new_sequence();
        action_lstm.start_new_sequence();
        // variables in the computation graph representing the parameters
        Expression pbias = parameter(*hg, p_pbias);
        Expression H = parameter(*hg, p_H);
        Expression D = parameter(*hg, p_D);
        Expression R = parameter(*hg, p_R);
        Expression cbias = parameter(*hg, p_cbias);
        Expression S = parameter(*hg, p_S);
        Expression B = parameter(*hg, p_B);
        Expression O = parameter(*hg, p_O);

        Expression A = parameter(*hg, p_A);
        Expression ib = parameter(*hg, p_ib);
        Expression w2l = parameter(*hg, p_w2l);
        Expression s2l = parameter(*hg, s_w2l);
        Expression p2l;
        if (USE_POS)
            p2l = parameter(*hg, p_p2l);
        Expression g2l = parameter(*hg, p_g2l);
        Expression t2l;
        if (p_t2l)
            t2l = parameter(*hg, p_t2l);
        Expression p2a = parameter(*hg, p_p2a);
        Expression abias = parameter(*hg, p_abias);
        Expression action_start = parameter(*hg, p_action_start);

        action_lstm.add_input(action_start);

        Expression cW = parameter(*hg, p_cW);

        vector<Expression> buffer(
                sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
        vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
        // precompute buffer representation from left to right


        Expression word_end = parameter(*hg, p_end_of_word); //Miguel
        Expression word_start = parameter(*hg, p_start_of_word); //Miguel
        fw_char_lstm.new_graph(*hg);
        bw_char_lstm.new_graph(*hg);
        for (unsigned i = 0; i < sent.size(); ++i) {
            assert(sent[i] < VOCAB_SIZE);
            std::string ww = str_sent[i];
            Expression w;
            Expression c;
            /**********SPELLING MODEL*****************/
            if (USE_SPELLING) {
                //std::cout<<"using spelling"<<"\n";
                if (ww.length() == 4 && ww[0] == 'R' && ww[1] == 'O' && ww[2] == 'O' && ww[3] == 'T') {
                    c = lookup(*hg, p_w,
                               sent[i]); //we do not need a LSTM encoding for the root word, so we put it directly-.
                }
                else {
                    fw_char_lstm.start_new_sequence();
                    fw_char_lstm.add_input(word_start);
                    std::vector<int> strevbuffer;
                    for (unsigned j = 0; j < ww.length(); j += UTF8Len(ww[j])) {
                        std::string wj;
                        for (unsigned h = j; h < j + UTF8Len(ww[j]); h++) wj += ww[h];
                        int wjint = corpus.charsToInt[wj];
                        strevbuffer.push_back(wjint);
                        Expression cj = lookup(*hg, char_emb, wjint);
                        fw_char_lstm.add_input(cj);
                    }
                    fw_char_lstm.add_input(word_end);
                    Expression fw_i = fw_char_lstm.back();
                    bw_char_lstm.start_new_sequence();
                    bw_char_lstm.add_input(word_end);
                    while (!strevbuffer.empty()) {
                        int wjint = strevbuffer.back();
                        Expression cj = lookup(*hg, char_emb, wjint);
                        bw_char_lstm.add_input(cj);
                        strevbuffer.pop_back();
                    }
                    bw_char_lstm.add_input(word_start);
                    Expression bw_i = bw_char_lstm.back();
                    vector<Expression> tt = {fw_i, bw_i};
                    c = concatenate(tt); //and this goes into the buffer...
                }
            }
            /**************************************************/
            //cerr<<"concatenate?"<<"\n";

            /***************NO SPELLING*************************************/

            // Expression w = lookup(*hg, p_w, sent[i]);
            w = lookup(*hg, p_w, sent[i]);
            Expression i_i;
            if (USE_SPELLING) {
                Expression p = lookup(*hg, p_p, sentPos[i]);
                i_i = affine_transform({ib, s2l, c, w2l, w, p2l, p});
            } else {
                Expression p = lookup(*hg, p_p, sentPos[i]);
                i_i = affine_transform({ib, w2l, w, p2l, p});
            }
            if (p_t && pretrained.count(raw_sent[i])) {
                Expression t = const_lookup(*hg, p_t, raw_sent[i]);
                i_i = affine_transform({i_i, t2l, t});
            }
            Expression gen_fea = cnn::expr::input(*hg, {GEN_DIM}, &current_gen_feature[i]);
            i_i = affine_transform({i_i, g2l, gen_fea});

            buffer[sent.size() - i] = rectify(i_i);
            bufferi[sent.size() - i] = i;
        }
        // dummy symbol to represent the empty buffer
        buffer[0] = parameter(*hg, p_buffer_guard);
        bufferi[0] = -999;
        for (auto &b : buffer)
            buffer_lstm.add_input(b);

        vector<Expression> stack;  // variables representing subtree embeddings
        vector<int> stacki; // position of words in the sentence of head of subtree
        stack.push_back(parameter(*hg, p_stack_guard));
        stacki.push_back(-999); // not used for anything

        vector<Expression> output;  // variables representing subtree embeddings
        vector<int> outputi;
        output.push_back(parameter(*hg, p_output_guard));
        outputi.push_back(-999); // not used for anything
        // drive dummy symbol on stack through LSTM
        stack_lstm.add_input(stack.back());
        output_lstm.add_input(output.back());
        vector<Expression> path_score; //the score for every path in the beam
        //string rootword;
        //unsigned action_count = 0;  // incremented at each prediction
        //while (buffer.size() > 1 || stack.size() > 1) {


        init.stack_lstm = stack_lstm;
        init.buffer_lstm = buffer_lstm;
        init.output_lstm = output_lstm;
        init.action_lstm = action_lstm;
        init.buffer = buffer;
        init.bufferi = bufferi;
        init.stack = stack;
        init.stacki = stacki;
        init.output = output;
        init.outputi = outputi;
        init.results = results;
        init.score = 0.0;
        init.is_gold = true;
        if (init.stacki.size() == 1 && init.bufferi.size() == 1) { assert(!"bad0"); }

        vector<ParserState> pq;
        pq.push_back(init);
        vector<ParserState> pq_new;
        vector<ParserState> pq_temp;
        pq_temp = pq;

        for (unsigned action_count = 0; action_count < sent.size(); ++action_count) {
            unsigned right_action = correct_actions[action_count];
            if (action_count > 0) {
                pq = pq_new;
                //pq_temp = pq_new;
            }
            pq_new.clear();
            while (pq.size() > 0) {
                const ParserState cur = pq.back();
                pq.pop_back();
                vector<unsigned> current_valid_actions;
                for (auto a: possible_actions) {
//                if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
//                    continue;
                    current_valid_actions.push_back(a);
                }
                Expression p_t = affine_transform(
                        {pbias, O, cur.output_lstm.back(), S, cur.stack_lstm.back(), B, cur.buffer_lstm.back(), A,
                         cur.action_lstm.back()});
                Expression nlp_t = rectify(p_t);
                // r_t = abias + p2a * nlp
                Expression adiste = affine_transform({abias, p2a, nlp_t});
                vector<float> adist = as_vector(hg->incremental_forward());

                //vector<float> adist = as_vector(hg->get_value(adiste));
                for (auto action : current_valid_actions) {
                    pq_new.resize(pq_new.size() + 1);
                    ParserState &ns = pq_new.back();
                    ns = cur;
                    ns.score += adist[action];
                    ns.results.push_back(action);
                    if ((ns.is_gold == true) && action == right_action) {
                        ns.is_gold = true;
                    }
                    else {
                        ns.is_gold = false;
                    }
                    if (action_count > 0) {
                        ns.score_exp = ns.score_exp + pick(adiste, action);
                    }
                    else {
                        ns.score_exp = pick(adiste, action);
                    }
                    Expression actione = lookup(*hg, p_a, action);
                    ns.action_lstm.add_input(actione);
                    const string &actionString = setOfActions[action];
                    //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
                    const char ac = actionString[0];
                    if (ac == 'S') {  // SHIFT
                        assert(ns.buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
                        ns.stack.push_back(ns.buffer.back());
                        ns.stack_lstm.add_input(ns.buffer.back());
                        ns.stacki.push_back(ns.bufferi.back());
                        ns.buffer.pop_back();
                        ns.buffer_lstm.rewind_one_step();
                        ns.bufferi.pop_back();
                    }
                    else if (ac == 'O') {
                        assert(ns.bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                        ns.outputi.push_back(ns.bufferi.back());
                        ns.output.push_back(ns.buffer.back());
                        ns.output_lstm.add_input(ns.buffer.back());
                        ns.buffer.pop_back();
                        ns.bufferi.pop_back();
                        ns.buffer_lstm.rewind_one_step();

                        while (ns.stacki.size() > 1) {
                            ns.stack_lstm.rewind_one_step();
                            ns.stack.pop_back();
                            ns.stacki.pop_back();
                        }
                    }
                }
            }
            pq_temp = pq_new;
            prune(pq_new, BEAM_SIZE);
            bool gold_state = false;
            for (unsigned j = 0; j < pq_new.size(); ++j) {
                if (pq_new[j].is_gold == true) {
                    gold_state = true;
                }
            }
            if (gold_state == false) {
                break;
            }
//            else {
//                pq_temp = pq_new;
//            }
        }

//        std::vector<Expression> f;
//        for (unsigned t = 0; t < n_labels; ++t) {
//            f.push_back(alpha[len - 1][t]);
//        }
//        return cnn::expr::logsumexp(f) - path.back();
        unsigned gold_position = 0;
        for (unsigned m = 0; m < pq_temp.size(); ++m) {
            if (pq_temp[m].is_gold == true) {
                gold_position = m;
            }
        }
        std::vector<Expression> f;
        for (unsigned t = 0; t < pq_temp.size(); ++t) {
            f.push_back(pq_temp[t].score_exp);
        }

        Expression tot_neglogprob = cnn::expr::logsumexp(f) - pq_temp[gold_position].score_exp;
        // Expression tot_neglogprob = -sum(log_probs);
        //assert(tot_neglogprob.pg != nullptr);
        return results;
//    }

    }


    vector<unsigned> log_prob_parser_beam_test(ComputationGraph *hg,
                                               const std::vector<std::vector<cnn::real>> current_gen_feature,
                                               const vector<string> &str_sent,  // raw sentence
                                               const vector<unsigned> &raw_sent,  // raw sentence
                                               const vector<unsigned> &sent,  // sent with oovs replaced
                                               const vector<unsigned> &sentPos,
                                               const vector<unsigned> &correct_actions,
                                               const vector<string> &setOfActions,
                                               const map<unsigned, std::string> &intToWords,
                                               bool is_evaluation,
                                               double *right) {
        //for (unsigned i = 0; i < sent.size(); ++i) cerr << ' ' << intToWords.find(sent[i])->second;
        //cerr << endl;
        // vector<unsigned> predict;
        vector<unsigned> results;
        const bool build_training_graph = correct_actions.size() > 0;
        //std::cout<<"****************"<<"\n";
        bool apply_dropout = (DROPOUT && !is_evaluation);
        ParserState init;

        stack_lstm.new_graph(*hg);
        buffer_lstm.new_graph(*hg);
        output_lstm.new_graph(*hg);
        action_lstm.new_graph(*hg);

        ent_lstm_fwd.new_graph(*hg);
        ent_lstm_rev.new_graph(*hg);


        if (apply_dropout) {
            stack_lstm.set_dropout(DROPOUT);
            action_lstm.set_dropout(DROPOUT);
            buffer_lstm.set_dropout(DROPOUT);
            ent_lstm_fwd.set_dropout(DROPOUT);
            ent_lstm_rev.set_dropout(DROPOUT);
        } else {
            stack_lstm.disable_dropout();
            action_lstm.disable_dropout();
            buffer_lstm.disable_dropout();
            ent_lstm_fwd.disable_dropout();
            ent_lstm_rev.disable_dropout();
        }

        stack_lstm.start_new_sequence();
        buffer_lstm.start_new_sequence();
        output_lstm.start_new_sequence();
        action_lstm.start_new_sequence();
        // variables in the computation graph representing the parameters
        Expression pbias = parameter(*hg, p_pbias);
        Expression H = parameter(*hg, p_H);
        Expression D = parameter(*hg, p_D);
        Expression R = parameter(*hg, p_R);
        Expression cbias = parameter(*hg, p_cbias);
        Expression S = parameter(*hg, p_S);
        Expression B = parameter(*hg, p_B);
        Expression O = parameter(*hg, p_O);

        Expression A = parameter(*hg, p_A);
        Expression ib = parameter(*hg, p_ib);
        Expression w2l = parameter(*hg, p_w2l);
        Expression s2l = parameter(*hg, s_w2l);
        Expression p2l;
        if (USE_POS)
            p2l = parameter(*hg, p_p2l);
        Expression g2l = parameter(*hg, p_g2l);
        Expression t2l;
        if (p_t2l)
            t2l = parameter(*hg, p_t2l);
        Expression p2a = parameter(*hg, p_p2a);
        Expression abias = parameter(*hg, p_abias);
        Expression action_start = parameter(*hg, p_action_start);

        action_lstm.add_input(action_start);

        Expression cW = parameter(*hg, p_cW);

        vector<Expression> buffer(
                sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
        vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
        // precompute buffer representation from left to right


        Expression word_end = parameter(*hg, p_end_of_word); //Miguel
        Expression word_start = parameter(*hg, p_start_of_word); //Miguel
        fw_char_lstm.new_graph(*hg);
        bw_char_lstm.new_graph(*hg);
        for (unsigned i = 0; i < sent.size(); ++i) {
            assert(sent[i] < VOCAB_SIZE);
            std::string ww = str_sent[i];
            Expression w;
            Expression c;
            /**********SPELLING MODEL*****************/
            if (USE_SPELLING) {
                //std::cout<<"using spelling"<<"\n";
                if (ww.length() == 4 && ww[0] == 'R' && ww[1] == 'O' && ww[2] == 'O' && ww[3] == 'T') {
                    c = lookup(*hg, p_w,
                               sent[i]); //we do not need a LSTM encoding for the root word, so we put it directly-.
                }
                else {
                    fw_char_lstm.start_new_sequence();
                    fw_char_lstm.add_input(word_start);
                    std::vector<int> strevbuffer;
                    for (unsigned j = 0; j < ww.length(); j += UTF8Len(ww[j])) {
                        std::string wj;
                        for (unsigned h = j; h < j + UTF8Len(ww[j]); h++) wj += ww[h];
                        int wjint = corpus.charsToInt[wj];
                        strevbuffer.push_back(wjint);
                        Expression cj = lookup(*hg, char_emb, wjint);
                        fw_char_lstm.add_input(cj);
                    }
                    fw_char_lstm.add_input(word_end);
                    Expression fw_i = fw_char_lstm.back();
                    bw_char_lstm.start_new_sequence();
                    bw_char_lstm.add_input(word_end);
                    while (!strevbuffer.empty()) {
                        int wjint = strevbuffer.back();
                        Expression cj = lookup(*hg, char_emb, wjint);
                        bw_char_lstm.add_input(cj);
                        strevbuffer.pop_back();
                    }
                    bw_char_lstm.add_input(word_start);
                    Expression bw_i = bw_char_lstm.back();
                    vector<Expression> tt = {fw_i, bw_i};
                    c = concatenate(tt); //and this goes into the buffer...
                }
            }
            /**************************************************/
            //cerr<<"concatenate?"<<"\n";

            /***************NO SPELLING*************************************/

            // Expression w = lookup(*hg, p_w, sent[i]);
            w = lookup(*hg, p_w, sent[i]);
            Expression i_i;
            if (USE_SPELLING) {
                Expression p = lookup(*hg, p_p, sentPos[i]);
                i_i = affine_transform({ib, s2l, c, w2l, w, p2l, p});
            } else {
                Expression p = lookup(*hg, p_p, sentPos[i]);
                i_i = affine_transform({ib, w2l, w, p2l, p});
            }
            if (p_t && pretrained.count(raw_sent[i])) {
                Expression t = const_lookup(*hg, p_t, raw_sent[i]);
                i_i = affine_transform({i_i, t2l, t});
            }
            Expression gen_fea = cnn::expr::input(*hg, {GEN_DIM}, &current_gen_feature[i]);
            i_i = affine_transform({i_i, g2l, gen_fea});

            buffer[sent.size() - i] = rectify(i_i);
            bufferi[sent.size() - i] = i;
        }
        // dummy symbol to represent the empty buffer
        buffer[0] = parameter(*hg, p_buffer_guard);
        bufferi[0] = -999;
        for (auto &b : buffer)
            buffer_lstm.add_input(b);

        vector<Expression> stack;  // variables representing subtree embeddings
        vector<int> stacki; // position of words in the sentence of head of subtree
        stack.push_back(parameter(*hg, p_stack_guard));
        stacki.push_back(-999); // not used for anything

        vector<Expression> output;  // variables representing subtree embeddings
        vector<int> outputi;
        output.push_back(parameter(*hg, p_output_guard));
        outputi.push_back(-999); // not used for anything
        // drive dummy symbol on stack through LSTM
        stack_lstm.add_input(stack.back());
        output_lstm.add_input(output.back());
        vector<Expression> path_score; //the score for every path in the beam
        //string rootword;
        //unsigned action_count = 0;  // incremented at each prediction
        //while (buffer.size() > 1 || stack.size() > 1) {


        init.stack_lstm = stack_lstm;
        init.buffer_lstm = buffer_lstm;
        init.output_lstm = output_lstm;
        init.action_lstm = action_lstm;
        init.buffer = buffer;
        init.bufferi = bufferi;
        init.stack = stack;
        init.stacki = stacki;
        init.output = output;
        init.outputi = outputi;
        init.results = results;
        init.score = 0;
        init.is_gold = true;
        if (init.stacki.size() == 1 && init.bufferi.size() == 1) { assert(!"bad0"); }

        vector<ParserState> pq;
        pq.push_back(init);
        vector<ParserState> pq_new;
        // vector<ParserState> pq_temp;
        // pq_temp = pq;

        for (unsigned action_count = 0; action_count < sent.size(); ++action_count) {
           // unsigned right_action = correct_actions[action_count];
            if (action_count > 0) {
                pq = pq_new;
                //pq_temp = pq_new;
            }
            pq_new.clear();
            while (pq.size() > 0) {
                const ParserState cur = pq.back();
                pq.pop_back();
                vector<unsigned> current_valid_actions;
                for (auto a: possible_actions) {
//                if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
//                    continue;
                    current_valid_actions.push_back(a);
                }
                Expression p_t = affine_transform(
                        {pbias, O, cur.output_lstm.back(), S, cur.stack_lstm.back(), B, cur.buffer_lstm.back(), A,
                         cur.action_lstm.back()});
                Expression nlp_t = rectify(p_t);
                // r_t = abias + p2a * nlp
                Expression adiste = affine_transform({abias, p2a, nlp_t});
                vector<float> adist = as_vector(hg->incremental_forward());

                //vector<float> adist = as_vector(hg->get_value(adiste));
                for (auto action : current_valid_actions) {
                    pq_new.resize(pq_new.size() + 1);
                    ParserState &ns = pq_new.back();
                    ns = cur;
                    ns.score += adist[action];
                    ns.results.push_back(action);
//                    if ((ns.is_gold == true) && action == right_action) {
//                        ns.is_gold = true;
//                    }
//                    else {
//                        ns.is_gold = false;
//                    }
                    if (action_count > 0) {
                        ns.score_exp = ns.score_exp + pick(adiste, action);
                    }
                    else {
                        ns.score_exp = pick(adiste, action);
                    }
                    Expression actione = lookup(*hg, p_a, action);
                    ns.action_lstm.add_input(actione);
                    const string &actionString = setOfActions[action];
                    //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
                    const char ac = actionString[0];
                    if (ac == 'S') {  // SHIFT
                        assert(ns.buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
                        ns.stack.push_back(ns.buffer.back());
                        ns.stack_lstm.add_input(ns.buffer.back());
                        ns.stacki.push_back(ns.bufferi.back());
                        ns.buffer.pop_back();
                        ns.buffer_lstm.rewind_one_step();
                        ns.bufferi.pop_back();
                    }
                    else if (ac == 'O') {
                        assert(ns.bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
                        ns.outputi.push_back(ns.bufferi.back());
                        ns.output.push_back(ns.buffer.back());
                        ns.output_lstm.add_input(ns.buffer.back());
                        ns.buffer.pop_back();
                        ns.bufferi.pop_back();
                        ns.buffer_lstm.rewind_one_step();

                        while (ns.stacki.size() > 1) {
                            ns.stack_lstm.rewind_one_step();
                            ns.stack.pop_back();
                            ns.stacki.pop_back();
                        }
                    }
                }
            }
            prune(pq_new, BEAM_SIZE);
        }

        std::vector<Expression> f;
        for (unsigned t = 0; t < pq_new.size(); ++t) {
            f.push_back(pq_new[t].score_exp);
        }

        Expression tot_neglogprob = cnn::expr::logsumexp(f) - pq_new[pq_new.size() - 1].score_exp;

        unsigned gold_position = 0;
        double best_score = -1000.0;
        for (unsigned m = 0; m < pq_new.size(); ++m) {
            if (pq_new[m].score >  best_score) {
                gold_position = m;
                best_score = pq_new[m].score;
            }
        }


        // Expression tot_neglogprob = -sum(log_probs);
        //assert(tot_neglogprob.pg != nullptr);
        return pq_new[gold_position].results;
//    }

    }
};

    void signal_callback_handler(int /* signum */) {
        if (requested_stop) {
            cerr << "\nReceived SIGINT again, quitting.\n";
            _exit(1);
        }
        cerr << "\nReceived SIGINT terminating optimization early...\n";
        requested_stop = true;
    }

    double evaluate(const po::variables_map &conf, ParserBuilder &parser, const std::string &dev_result,
                    const std::set<unsigned> &training_vocab) {
        unsigned dev_size = corpus.nsentencesDev;
        ofstream ofs(dev_result);
        double right = 0;
        unsigned num_of_gold = 0;
        unsigned num_of_predict = 0;
        unsigned num_of_right = 0;
        const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
            const vector<unsigned> &sentence = corpus.sentencesDev[sii];
            const vector<unsigned> &sentencePos = corpus.sentencesPosDev[sii];
            const vector<unsigned> &actions = corpus.correct_act_sentDev[sii];
            const vector<string> &sentenceUnkStr = corpus.sentencesStrDev[sii];
            const std::vector<std::vector<cnn::real>> current_gen_feature = corpus.devel_general_feature[sii];
            vector<unsigned> tsentence = sentence;
            if (!USE_SPELLING) {
                for (auto &w : tsentence)
                    if (training_vocab.count(w) == 0) w = kUNK;
            }
            ComputationGraph hg1;
            vector<unsigned> pred = parser.log_prob_parser_beam_test(&hg1, current_gen_feature, sentenceUnkStr, sentence,
                                                                tsentence,
                                                                sentencePos,
                                                                vector<unsigned>(),
                                                                corpus.actions, corpus.intToWords, true, &right);
            map<int, string> ref = parser.compute_ents(sentence.size(), actions, corpus.actions);
            map<int, string> hyp = parser.compute_ents(sentence.size(), pred, corpus.actions);
            for (unsigned i = 0; i < sentenceUnkStr.size(); ++i) {
                ofs << sentenceUnkStr[i] << " " << corpus.intToPos[sentencePos[i]] << " " << ref.find(i)->second <<
                " " <<
                hyp.find(i)->second << std::endl;
            }
            ofs << std::endl;
            for (unsigned i = 0; i < sentence.size(); ++i) {
                auto ri = ref.find(i);
                auto hi = hyp.find(i);
                if (ri->second[0] == 'R') {
                    num_of_gold += 1;
                }
                if (hi->second[0] == 'R') {
                    num_of_predict += 1;
                }
                if (ri->second[0] == 'R' && hi->second[0] == 'R') {
                    num_of_right += 1;
                }
            }
        }
        ofs.close();
        double global_f1_score = 0;
        global_f1_score = num_of_right * 2.0 / (num_of_predict + num_of_gold);
        _INFO << "Dev p-score: " << num_of_right * 1.0 / num_of_predict;
        _INFO << "Dev R-score: " << num_of_right * 1.0 / num_of_gold;
        _INFO << "Dev F-score: " << global_f1_score << endl;
        return global_f1_score;
        //auto t_end = std::chrono::high_resolution_clock::now();
        //cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " f1: " << global_f1_score << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
    }


    double evaluate_test(const po::variables_map &conf, ParserBuilder &parser, const std::string &test_result,
                         const std::set<unsigned> &training_vocab) {
        unsigned test_size = corpus.nsentencesTest;
        ofstream ofs(test_result);
        double right = 0;
        unsigned num_of_gold = 0;
        unsigned num_of_predict = 0;
        unsigned num_of_right = 0;
        const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
            const vector<unsigned> &sentence = corpus.sentencesTest[sii];
            const vector<unsigned> &sentencePos = corpus.sentencesPosTest[sii];
            const vector<unsigned> &actions = corpus.correct_act_sentTest[sii];
            const vector<string> &sentenceUnkStr = corpus.sentencesStrTest[sii];
            const std::vector<std::vector<cnn::real>> current_gen_feature = corpus.test_general_feature[sii];
            vector<unsigned> tsentence = sentence;
            if (!USE_SPELLING) {
                for (auto &w : tsentence)
                    if (training_vocab.count(w) == 0) w = kUNK;
            }
            ComputationGraph hg1;
            vector<unsigned> pred = parser.log_prob_parser_beam_test(&hg1, current_gen_feature, sentenceUnkStr, sentence,
                                                                tsentence,
                                                                sentencePos,
                                                                vector<unsigned>(),
                                                                corpus.actions, corpus.intToWords, true, &right);
            map<int, string> ref = parser.compute_ents(sentence.size(), actions, corpus.actions);
            map<int, string> hyp = parser.compute_ents(sentence.size(), pred, corpus.actions);
            for (unsigned i = 0; i < sentenceUnkStr.size(); ++i) {
                ofs << sentenceUnkStr[i] << " " << corpus.intToPos[sentencePos[i]] << " " << ref.find(i)->second <<
                " " <<
                hyp.find(i)->second << std::endl;
            }
            ofs << std::endl;
            for (unsigned i = 0; i < sentence.size(); ++i) {
                auto ri = ref.find(i);
                auto hi = hyp.find(i);
                if (ri->second[0] == 'R') {
                    num_of_gold += 1;
                }
                if (hi->second[0] == 'R') {
                    num_of_predict += 1;
                }
                if (ri->second[0] == 'R' && hi->second[0] == 'R') {
                    num_of_right += 1;
                }
            }
        }
        ofs.close();
        double global_f1_score = 0;
        global_f1_score = num_of_right * 2.0 / (num_of_predict + num_of_gold);
        _INFO << "TEST p-score: " << num_of_right * 1.0 / num_of_predict;
        _INFO << "TEST R-score: " << num_of_right * 1.0 / num_of_gold;
        _INFO << "TEST F-score: " << global_f1_score << endl;
        return global_f1_score;
        //auto t_end = std::chrono::high_resolution_clock::now();
        //cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " f1: " << global_f1_score << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
    }


    int main(int argc, char **argv) {
        cnn::Initialize(argc, argv);

        cerr << "COMMAND:";
        for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
        cerr << endl;
        unsigned status_every_i_iterations = 100;

        po::variables_map conf;
        InitCommandLine(argc, argv, &conf);
        USE_POS = conf.count("use_pos_tags");
        if (conf.count("dropout"))
            DROPOUT = conf["dropout"].as<float>();
        USE_SPELLING = conf.count("use_spelling"); //Miguel
        corpus.USE_SPELLING = USE_SPELLING;
        LAYERS = conf["layers"].as<unsigned>();
        WORD_DIM = conf["word_dim"].as<unsigned>();
        CHAR_DIM = conf["char_dim"].as<unsigned>();
        PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
        HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
        ACTION_DIM = conf["action_dim"].as<unsigned>();
        GEN_DIM = conf["gen_fea_dim"].as<unsigned>();
        LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
        POS_DIM = conf["pos_dim"].as<unsigned>();
        REL_DIM = conf["rel_dim"].as<unsigned>();
        BEAM_SIZE = conf["beam_size"].as<unsigned>();

        const unsigned beam_size = conf["beam_size"].as<unsigned>();
        const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
        cerr << "Unknown word strategy: ";
        if (unk_strategy == 1) {
            cerr << "STOCHASTIC REPLACEMENT\n";
        } else {
            abort();
        }
        const double unk_prob = conf["unk_prob"].as<double>();
        assert(unk_prob >= 0.);
        assert(unk_prob <= 1.);
        ostringstream os;
        os << "parser_" << (USE_POS ? "pos" : "nopos")
        << '_' << LAYERS
        << '_' << WORD_DIM
        << '_' << HIDDEN_DIM
        << '_' << ACTION_DIM
        << '_' << LSTM_INPUT_DIM
        << '_' << POS_DIM
        << '_' << REL_DIM
        << "_d" << DROPOUT
        << "-pid" << getpid() << ".params";



        int best_correct_heads = 0;
        double best_f1_score = -1.0;
        double best_test_score = -1.0;
        const string fname = os.str();
        cerr << "Writing parameters to file: " << fname << endl;
        //bool softlinkCreated = false;
        corpus.load_correct_actions(conf["training_data"].as<string>(), GEN_DIM);
        const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
        kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);

        if (conf.count("words")) {
            pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
            cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
            ifstream in(conf["words"].as<string>().c_str());
            string line;
            getline(in, line);
            vector<float> v(PRETRAINED_DIM, 0);
            string word;
            while (getline(in, line)) {
                istringstream lin(line);
                lin >> word;
                for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
                unsigned id = corpus.get_or_add_word(word);
                pretrained[id] = v;
            }
        }

        set<unsigned> training_vocab; // words available in the training corpus
        set<unsigned> singletons;
        {  // compute the singletons in the parser's training data
            map<unsigned, unsigned> counts;
            for (auto sent : corpus.sentences)
                for (auto word : sent.second) {
                    training_vocab.insert(word);
                    counts[word]++;
                }
            for (auto wc : counts)
                if (wc.second == 1) singletons.insert(wc.first);
        }

        cerr << "Number of words: " << corpus.nwords << endl;
        VOCAB_SIZE = corpus.nwords + 1;

        cerr << "Number of UTF8 chars: " << corpus.maxChars << endl;
        if (corpus.maxChars > 255) CHAR_SIZE = corpus.maxChars;

        ACTION_SIZE = corpus.nactions + 1;
        //POS_SIZE = corpus.npos + 1;
        POS_SIZE = corpus.npos + 10;
        possible_actions.resize(corpus.nactions);
        for (unsigned i = 0; i < corpus.nactions; ++i)
            possible_actions[i] = i;

        Model model;
        ParserBuilder parser(&model, pretrained);
        if (conf.count("model")) {
            ifstream in(conf["model"].as<string>().c_str());
            boost::archive::text_iarchive ia(in);
            ia >> model;
        }
        // OOV words will be replaced by UNK tokens
        cerr << "tag0" << endl;
        corpus.load_correct_actionsDev(conf["dev_data"].as<string>(), GEN_DIM);
        corpus.load_correct_actionsTest(conf["test_data"].as<string>(), GEN_DIM);
        cerr << "tag1" << endl;
        if (USE_SPELLING) VOCAB_SIZE = corpus.nwords + 1;


        //TRAINING
        if (conf.count("train")) {
            // signal(SIGINT, signal_callback_handler);
            //SimpleSGDTrainer sgd(&model);
            cnn::Trainer *trainer = get_trainer(conf, &model);
            //MomentumSGDTrainer sgd(&model);
            //sgd.eta_decay = 0.08;
            //sgd.eta_decay = 0.05;
            cerr << "Training started." << "\n";
            vector<unsigned> order(corpus.nsentences);
            for (unsigned i = 0; i < corpus.nsentences; ++i)
                order[i] = i;
            double tot_seen = 0;
//            cerr << "tag2" << endl;
            status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
            //unsigned si = corpus.nsentences;
            cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
            double right = 0;
            double llh = 0;
            bool first = true;
            //unsigned iter = 0;
            unsigned maxiter = conf["maxiter"].as<unsigned>();
            unsigned logc = 0;

            std::stringstream best_model_ss;
            double n_corr_tokens = 0;
            double batchly_n_corr_tokens = 0, batchly_n_tokens = 0, batchly_llh = 0;
            float eta_update = conf["eta_update"].as<float>();
            for (unsigned iter = 0; iter < maxiter; ++iter) {
                _INFO << "start of iteration #" << iter << ", training data is shuffled.";
                random_shuffle(order.begin(), order.end());
                for (unsigned i = 0; i < order.size(); ++i) {
                    auto si = order[i];
                    const vector<unsigned> &sentence = corpus.sentences[si];
                    vector<unsigned> tsentence = sentence;
                    const std::vector<std::vector<cnn::real>> current_gen_feature = corpus.train_general_feature[si];
                    if (unk_strategy == 1) {
                        for (auto &w : tsentence)
                            if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
                    }
                    const vector<unsigned> &sentencePos = corpus.sentencesPos[si];
                    const vector<unsigned> &actions = corpus.correct_act_sent[si];
                    const vector<string> &sentenceUnkStr = corpus.sentencesStrTrain[si];
                    double lp;
                    {
                        ComputationGraph hg;
                        parser.log_prob_parser_beam_train(&hg, current_gen_feature, sentenceUnkStr, sentence, tsentence,
                                                          sentencePos,
                                                          actions,
                                                          corpus.actions,
                                                          corpus.intToWords,
                                                          false, &right);
                        lp = as_scalar(hg.incremental_forward());
                        if (lp < 0) {
                            cerr << "Log prob < 0 on sentence " << si << ": lp=" << lp << endl;
                            assert(lp >= 0.0);
                        }
                        hg.backward();
                        trainer->update(eta_update);
                    }
                    tot_seen += 1;
                    llh += lp;
                    batchly_llh += lp;
                    batchly_n_tokens += actions.size();
                    ++si;
                    ++logc;
                    if (logc % conf["report_stops"].as<unsigned>() == 0) {
                        trainer->status();
                        cerr << "iter (batch) #" << iter << " (epoch " << tot_seen / corpus.nsentences
                        << ") llh: " << batchly_llh << " ppl: " << exp(batchly_llh / batchly_n_tokens) << endl;
                        n_corr_tokens += batchly_n_corr_tokens;
                        batchly_llh = batchly_n_tokens = batchly_n_corr_tokens = 0.;
                    }
                    if (logc % conf["evaluate_stops"].as<unsigned>() == 0 && iter <4) {
                        string dev_result = conf["dev_result"].as<string>();
                        string test_result = conf["test_result"].as<string>();
                        double global_f1_score = evaluate(conf, parser, dev_result, training_vocab);
                       // double global_test_score = evaluate_test(conf, parser, test_result, training_vocab);
                        if (global_f1_score > best_f1_score) {
                            best_f1_score = global_f1_score;
                            _INFO << "new dev best record " << best_f1_score << " is achieved, model updated.";
                            best_model_ss.str("");
                            boost::archive::text_oarchive oa(best_model_ss);
                            oa << model;
                        }
                    }
                    else if (logc % conf["evaluate_stops"].as<unsigned>() == 0 && iter >= 4) {
                        string dev_result = conf["dev_result"].as<string>();
                        string test_result = conf["test_result"].as<string>();
                        double global_f1_score = evaluate(conf, parser, dev_result, training_vocab);
                        double global_test_score = evaluate_test(conf, parser, test_result, training_vocab);
                        if (global_f1_score > best_f1_score) {
                            best_f1_score = global_f1_score;
                            _INFO << "new dev best record " << best_f1_score << " is achieved, model updated.";
                            best_model_ss.str("");
                            boost::archive::text_oarchive oa(best_model_ss);
                            oa << model;
                        }
                        if (global_test_score > best_test_score) {
                            best_test_score = global_test_score;
                            _INFO << "new test best record " << best_test_score << " is achieved, model updated.";
                        }
                    }
                }
                if (conf["optimizer"].as<std::string>() == "simple_sgd" ||
                    conf["optimizer"].as<std::string>() == "momentum_sgd") {
                    trainer->update_epoch();
                }
            }

            std::ofstream out(fname);
            boost::archive::text_iarchive ia(best_model_ss);
            ia >> model;
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
        } // should do training?








        if (true) { // do test evaluation
            string test_result = conf["test_result"].as<string>();
            string dev_result = conf["dev_result"].as<string>();
            evaluate(conf, parser, dev_result, training_vocab);
            evaluate_test(conf, parser, test_result, training_vocab);
        }
    }






























































