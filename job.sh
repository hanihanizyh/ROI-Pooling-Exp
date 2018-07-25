#!/bin/bash
cd ${TMPDIR}
ls ./
uname -a
date
env
date
CWD=`pwd`
echo ${CWD}
export PYTHONPATH=${CWD}:$PYTHONPATH
export CLASSPATH=$HADOOP_PREFIX/lib/classpath_hdfs.jar
export JAVA_TOOL_OPTIONS="-Xms2000m -Xmx10000m"
#HADOOP_JOB_HOME=/opt/hdfs/open_mlp/run_data/output/${job_name}/log/

CMD="python ./train_end2end.py --gpu 0,1"
echo Running ${CMD}
${CMD}
CMD="python test.py"
echo Running ${CMD}
${CMD}
