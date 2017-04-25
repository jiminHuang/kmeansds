#!/bin/sh
# 
# Author: jimin.huang
# 

# Parameters
workDir="../mimlre-2012-11-27"
dataDir="$workDir/corpora/kbp/train"
outputDir="$workDir/corpora/kbp/train/genTest"
clusterNumber=100
negativeRatio=5
subsample="True"
fileNum=88
negatives="$workDir/corpora/kbp/train/genTest/*.negatives"
model="mimlre"
modelFile="$workDir/corpora/kbp/kbp_relation_model_$model.*"
exeTimes=1

# Create result dir if not exists
if [ ! -d result ];then
    mkdir result
fi

for index in $(seq $exeTimes); do
    # Create present result dir according to time
    presentResultDir=result/`date +%Y%m%d-%H%M%S`-$model
    if [ ! -d $presentResultDir ];then
        mkdir $presentResultDir
    fi

    rm $modelFile
    rm $negatives

    python clustered_ds.py -n $clusterNumber -r $negativeRatio -o $outputDir -d $dataDir -s $subsample -f $fileNum 2> $presentResultDir/cluster_result

    #cd $workDir
    #./run.sh edu.stanford.nlp.kbp.slotfilling.KBPTrainer -props config/kbp/kbp_${model}.properties 1>$OLDPWD/$presentResultDir/runtime.log

    #mv corpora/kbp/${model}.* $OLDPWD/$presentResultDir/
    #cd $OLDPWD
done;
