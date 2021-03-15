from __future__ import print_function
import sys
from random import random
from operator import add

from pyspark.sql import SparkSession
from pyspark import SparkFiles
import os
from os import listdir


def score(jobInput, jobOutput, modelAssets):
    print("this is Nicky")
    print("Hello from score function: estimatePi")

    spark = SparkSession.builder.appName("DriftTest").getOrCreate()

    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = 100000 * partitions

    def f(_):
        x = random() * 2 - 1
        y = random() * 2 - 1
        return 1 if x ** 2 + y ** 2 <= 1 else 0

    count = (
        spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    )
    print("Pi is roughly %f" % (4.0 * count / n))

    spark.stop()


def metrics(external_inputs, external_outputs, external_model_assets):
    spark = SparkSession.builder.appName("DriftTest").getOrCreate()

    spark.read.csv(xternal_inputs["inputData"][0]["fileURL"])

    print("args ===")
    print(external_inputs, external_outputs, external_model_assets)
    print("===")

    print("output here: hi")
    # TODO: write to hadoop

    spark.stop()
