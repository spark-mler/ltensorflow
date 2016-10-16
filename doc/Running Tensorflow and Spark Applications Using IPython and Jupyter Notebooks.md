[Running Spark Applications Using IPython and Jupyter Notebooks](http://www.cloudera.com/documentation/enterprise/5-5-x/topics/spark_ipython.html)

replace pip by conda and install ipykernel:
```bash
conda install ipykernel
```

Starting a Notebook with PySpark:
```bash
cd  ~/wangyuming/data/jupyter
export PATH=/opt/cloudera/parcels/Anaconda/bin/:$PATH
export PYSPARK_DRIVER_PYTHON=/opt/cloudera/parcels/Anaconda/bin/jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --NotebookApp.open_browser=False --NotebookApp.ip='*' --NotebookApp.port=4059"
export PYSPARK_PYTHON="/opt/cloudera/parcels/Anaconda/bin/python"
pyspark --total-executor-cores 100 --executor-memory 4G --driver-memory 4G
```