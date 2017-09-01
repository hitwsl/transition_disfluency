INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disfluency-beam lstm-parse.cc ../training_utils.cc ${SHARED_SRC})
target_link_libraries(disfluency-beam cnn ${Boost_LIBRARIES})
