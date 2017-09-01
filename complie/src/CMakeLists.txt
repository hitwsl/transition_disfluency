set(SHARED_SRC
    ${source_directory}/training_utils.cc
    ${source_directory}/utils.cc
    ${source_directory}/layer.cc
    ${source_directory}/logging.cc
    ${source_directory}/labeling_deletion_model.cc
    ${source_directory}/corpus.cc)

#add_subdirectory(parser)
add_subdirectory(disfluency)
#add_subdirectory(rnnlm)

