cmake_minimum_required(VERSION 2.8)

project(insulator-diag)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV 4 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(include)

set(BOWKMajority src/BOWKmajorityTrainer.cpp include/BOWKmajorityTrainer.h)

add_executable(check_akaze_kpts src/check_akaze_kpts.cpp)
target_link_libraries(check_akaze_kpts ${OpenCV_LIBS})

add_executable(check_sift_kpts src/check_sift_kpts.cpp)
target_link_libraries(check_sift_kpts ${OpenCV_LIBS})

add_executable(check_agast_kpts src/check_agast_kpts.cpp)
target_link_libraries(check_agast_kpts ${OpenCV_LIBS})

add_executable(collect_desc src/collect_desc.cpp)
target_link_libraries(collect_desc ${OpenCV_LIBS} stdc++fs)

add_executable(train_vocab src/train_vocab.cpp ${BOWKMajority})
target_link_libraries(train_vocab ${OpenCV_LIBS})

add_executable(train_vocab_dist src/train_vocab_dist.cpp ${BOWKMajority})
target_link_libraries(train_vocab_dist ${OpenCV_LIBS})

add_executable(train_svm src/train_svm.cpp)
target_link_libraries(train_svm ${OpenCV_LIBS} stdc++fs)

add_executable(distance_minmax src/distance-minmax.cpp)
target_link_libraries(distance_minmax ${OpenCV_LIBS} stdc++fs)

add_executable(descriptor-filter src/descriptor-filter.cpp)
target_link_libraries(descriptor-filter ${OpenCV_LIBS})

add_executable(descriptor-filter-ratio-test src/descriptor-filter-ratio-test.cpp)
target_link_libraries(descriptor-filter-ratio-test ${OpenCV_LIBS})

add_executable(descriptor-filter-crosscheck src/descriptor-filter-crosscheck.cpp)
target_link_libraries(descriptor-filter-crosscheck ${OpenCV_LIBS})

add_executable(test_detector src/test_detector.cpp)
target_link_libraries(test_detector ${OpenCV_LIBS} stdc++fs)

add_executable(predict src/predict.cpp)
target_link_libraries(predict ${OpenCV_LIBS})