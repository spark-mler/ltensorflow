# Install tensorflow

### Our cluster environment:

CentOS 6.7, JDK 1.8, Cloudera Manager 5.5.1, CDH 5.4.3, Anaconda 4.0.0, Spark 2.0.0([install latest spark](https://github.com/wangyum/cm_csds/tree/master/SPARK)),

### Upgrade gcc to 4.8.2
```bash
wget http://people.centos.org/tru/devtools-2/devtools-2.repo -O /etc/yum.repos.d/devtools-2.repo
yum install devtoolset-2-gcc devtoolset-2-binutils devtoolset-2-gcc-c++
# active devtoolset-2
scl enable devtoolset-2 bash
# verification
gcc -v
```

### Install bazel:
```bash
git clone https://github.com/bazelbuild/bazel.git
cd bazel
./compile.sh
```
### Configure the installation
```
export PATH=/opt/cloudera/parcels/Anaconda/bin/:/root/wangyuming/bazel/output/:${PATH}

git clone https://github.com/tensorflow/tensorflow
git checkout r0.9

# ./configure
Please specify the location of python. [Default is /opt/cloudera/parcels/Anaconda/bin/python]:
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with GPU support? [y/N] N
No GPU support will be enabled for TensorFlow
Configuration finished
```

### Package tensorflow
```bash
/root/opensource/bazel/output/bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

### Install tensorflow
```bash
pip install --ignore-installed --upgrade  /tmp/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl
```

### Known issues
[`undefined symbol: clock_gettime`](https://github.com/tensorflow/tensorflow/issues/121)
