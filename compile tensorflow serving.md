```
[root@10-11-51-74 bazel]# bash ./compile.sh
INFO: You can skip this first step by providing a path to the bazel binary as second argument:
INFO:    ./compile.sh compile /path/to/bazel
   Building Bazel from scratch
ERROR: Must specify PROTOC if not bootstrapping from the distribution artifact; NOTE: development versions are built with 'bazel build //src:bazel'
```
Download [this](https://github.com/bazelbuild/bazel/releases/download/0.4.2/bazel-0.4.2-dist.zip) to compile

```
ERROR: /data/wym/serving/tensorflow/tensorflow/python/BUILD:1728:1: in cc_library rule //tensorflow/python:tf_session_helper: non-test target '//tensorflow/python:tf_session_helper' depends on testonly target '//tensorflow/python:construction_fails_op' and doesn't have testonly attribute set.
ERROR: Analysis of target '//tensorflow/tools/pip_package:build_pip_package' failed; build aborted.
INFO: Elapsed time: 2.632s
```
Please refer https://github.com/tensorflow/tensorflow/pull/5144

```
undefined symbol: clock_gettime
```
bazel build --linkopt='-lrt'  tensorflow_serving/...
