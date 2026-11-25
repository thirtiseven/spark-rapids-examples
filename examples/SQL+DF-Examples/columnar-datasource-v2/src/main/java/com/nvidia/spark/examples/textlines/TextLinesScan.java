/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.examples.textlines;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.network.util.JavaUtils;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.util.SerializableConfiguration;

/** Splits the files like Hadoop's TextInputFormat and packs the splits into partitions. */
final class TextLinesScan implements Scan, Batch {
  // The defaults keep every Arrow buffer (including BigIntVector's combined data+validity buffer)
  // at or below 2 MiB; see the README.
  static final int DEFAULT_MAX_BATCH_ROWS = 1 << 17;
  static final String DEFAULT_MAX_BATCH_BYTES = "2m";

  private final CaseInsensitiveStringMap options;
  private final StructType required;
  private final int maxBatchRows;
  private final long maxBatchBytes;
  private Configuration hadoopConf;
  private InputPartition[] partitions;
  private PartitionReaderFactory readerFactory;

  TextLinesScan(CaseInsensitiveStringMap options, StructType required) {
    this.options = options;
    this.required = required;
    this.maxBatchRows = options.getInt("maxBatchRows", DEFAULT_MAX_BATCH_ROWS);
    this.maxBatchBytes = JavaUtils.byteStringAsBytes(
        options.getOrDefault("maxBatchBytes", DEFAULT_MAX_BATCH_BYTES));
    if (maxBatchRows <= 0) {
      throw new IllegalArgumentException("maxBatchRows must be positive, got " + maxBatchRows);
    }
    if (maxBatchBytes <= 0 || maxBatchBytes > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "maxBatchBytes must be in (0, " + Integer.MAX_VALUE + "], got " + maxBatchBytes);
    }
    if (options.containsKey("maxPartitionBytes")) {
      long configured = JavaUtils.byteStringAsBytes(options.get("maxPartitionBytes"));
      if (configured <= 0) {
        throw new IllegalArgumentException(
            "maxPartitionBytes must be positive, got " + configured);
      }
    }
  }

  @Override
  public StructType readSchema() {
    return required;
  }

  @Override
  public String description() {
    return "TextLinesScan(" + String.join(",", paths()) + ")";
  }

  @Override
  public ColumnarSupportMode columnarSupportMode() {
    return ColumnarSupportMode.SUPPORTED;
  }

  @Override
  public Batch toBatch() {
    return this;
  }

  private String[] paths() {
    List<String> result = new ArrayList<>();
    // DataFrameReader.load(paths: _*) passes several paths as a JSON array.
    String json = options.get("paths");
    if (json != null) {
      try {
        Collections.addAll(result, new ObjectMapper().readValue(json, String[].class));
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    String single = options.get("path");
    if (single != null) {
      result.add(single);
    }
    if (result.isEmpty()) {
      throw new IllegalArgumentException("textlines requires a path");
    }
    return result.toArray(new String[0]);
  }

  // Like Spark's file sources: the session's Hadoop settings, such as those set with
  // spark.conf.set, and the options of this read, such as fs.s3a.endpoint.
  private Configuration hadoopConf() {
    if (hadoopConf == null) {
      hadoopConf = SparkSession.active().sessionState().newHadoopConf();
      for (Map.Entry<String, String> e : options.asCaseSensitiveMap().entrySet()) {
        if (e.getValue() != null
            && !e.getKey().equalsIgnoreCase("path")
            && !e.getKey().equalsIgnoreCase("paths")) {
          hadoopConf.set(e.getKey(), e.getValue());
        }
      }
    }
    return hadoopConf;
  }

  // Spark may call these several times while planning; list and split the input only once.
  @Override
  public synchronized InputPartition[] planInputPartitions() {
    if (partitions == null) {
      partitions = computePartitions();
    }
    return partitions;
  }

  @Override
  public synchronized PartitionReaderFactory createReaderFactory() {
    if (readerFactory == null) {
      SparkSession spark = SparkSession.active();
      // SerializableConfiguration is registered with Spark's KryoSerializer out of the box.
      Broadcast<SerializableConfiguration> conf = JavaSparkContext.fromSparkContext(
          spark.sparkContext()).broadcast(new SerializableConfiguration(hadoopConf()));
      readerFactory = new TextLinesReaderFactory(conf, required, maxBatchRows, maxBatchBytes);
    }
    return readerFactory;
  }

  private InputPartition[] computePartitions() {
    SparkSession spark = SparkSession.active();
    boolean explicitMax = options.containsKey("maxPartitionBytes");
    long maxPartitionBytes = JavaUtils.byteStringAsBytes(explicitMax
        ? options.get("maxPartitionBytes")
        : spark.conf().get("spark.sql.files.maxPartitionBytes"));
    long openCost = JavaUtils.byteStringAsBytes(
        spark.conf().get("spark.sql.files.openCostInBytes"));
    if (maxPartitionBytes <= 0) {
      throw new IllegalArgumentException(
          "maxPartitionBytes must be positive, got " + maxPartitionBytes);
    }
    if (openCost < 0) {
      throw new IllegalArgumentException(
          "spark.sql.files.openCostInBytes must be non-negative, got " + openCost);
    }
    int parallelism = Math.max(1, spark.sparkContext().defaultParallelism());
    long[] targetBytes = {maxPartitionBytes};
    List<InputSplit> splits;
    try {
      Job job = Job.getInstance(new Configuration(hadoopConf()));
      for (String path : paths()) {
        FileInputFormat.addInputPath(job, new Path(path));
      }
      // TextInputFormat keeps compressed files that cannot be split in one piece.
      // Override listStatus and computeSplitSize so splitting is not fragmented by a small
      // filesystem block size (e.g. 32 MiB on LocalFS/S3A) and scales the default partition cap
      // by defaultParallelism like Spark's FilePartition.maxSplitBytes without listing twice.
      TextInputFormat format = new TextInputFormat() {
        @Override
        protected List<FileStatus> listStatus(JobContext context) throws IOException {
          List<FileStatus> files = super.listStatus(context);
          if (!explicitMax && !files.isEmpty()) {
            long totalBytes = 0;
            for (FileStatus f : files) {
              totalBytes += f.getLen() + openCost;
            }
            targetBytes[0] = Math.min(maxPartitionBytes,
                Math.max(1L, Math.max(openCost, totalBytes / parallelism)));
          }
          return files;
        }

        @Override
        protected long computeSplitSize(long blockSize, long minSize, long maxSize) {
          return Math.max(minSize, Math.min(maxSize, targetBytes[0]));
        }
      };
      splits = format.getSplits(job);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    // Like Spark's file sources, pack small splits into one partition, counting each split as
    // at least openCostInBytes, so that a task builds GPU batches of a useful size.
    List<InputPartition> result = new ArrayList<>();
    List<FileSplit> current = new ArrayList<>();
    long currentBytes = 0;
    for (InputSplit split : splits) {
      FileSplit fileSplit = (FileSplit) split;
      if (!current.isEmpty() && currentBytes + fileSplit.getLength() > targetBytes[0]) {
        result.add(TextSplitPartition.of(current));
        current = new ArrayList<>();
        currentBytes = 0;
      }
      current.add(fileSplit);
      currentBytes += fileSplit.getLength() + openCost;
    }
    if (!current.isEmpty()) {
      result.add(TextSplitPartition.of(current));
    }
    return result.toArray(new InputPartition[0]);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof TextLinesScan)) {
      return false;
    }
    TextLinesScan that = (TextLinesScan) o;
    return options.equals(that.options) && required.equals(that.required);
  }

  @Override
  public int hashCode() {
    return Objects.hash(options, required);
  }
}
