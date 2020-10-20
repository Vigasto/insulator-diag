# insulator-diag

## Requirements
- CMake >= 2.8
- OpenCV >= 4.0
- C/C++ compiler with c++17

## Build

```
$ makedir build && cd build
$ make ..
```
## Usage
```
// collect descriptors from dataset
$ ./collect_desc <positive_dir> <negative_dir> descriptors.yml

// cluster descriptors as BoW vocab
$ ./train_vocab descriptors.yml <n cluster> vocab.yml

// train svm based on detected BoWs 
$ ./train_svm vocab.yml svm_param <positive_dir> <negative_dir>

//test svm scene classifier 
./test_detector <positive_dir> <negative_dir> svm_param vocab.yml

// predict single image
./predict <image.jpg> svm_param vocab.yml
```