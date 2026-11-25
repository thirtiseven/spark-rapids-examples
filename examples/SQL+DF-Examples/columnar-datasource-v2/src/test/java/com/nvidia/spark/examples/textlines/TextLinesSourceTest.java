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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.sql.vectorized.ArrowColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class TextLinesSourceTest {
  private static final int LONG_LINES_PER_FILE = 19;
  private static SparkSession spark;
  private static File dir;
  private static final List<File> files = new ArrayList<>();

  @BeforeClass
  public static void setUp() throws IOException {
    spark = SparkSession.builder().master("local[2]").appName("TextLinesSourceTest")
        .config("spark.ui.enabled", "false").getOrCreate();
    dir = Files.createTempDirectory("textlines").toFile();
    // Empty lines, multi-byte characters, a long line and both line terminators
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < 20000; i++) {
      if (i % 97 == 0) {
        text.append("\n");
      } else if (i % 1001 == 0) {
        text.append(String.join("", Collections.nCopies(5000, "x"))).append("\r\n");
      } else {
        text.append("line ").append(i).append(" é中\n");
      }
    }
    for (int f = 0; f < 3; f++) {
      File file = new File(dir, "part-" + f + ".txt");
      Files.write(file.toPath(), (text + "last line without terminator " + f)
          .getBytes(StandardCharsets.UTF_8));
      files.add(file);
    }
  }

  @AfterClass
  public static void tearDown() {
    spark.stop();
    for (File f : files) {
      f.delete();
    }
    dir.delete();
  }

  // Small splits and batches, so lines cross split and batch boundaries.
  private DataFrameReader reader(boolean includeBytes) {
    return spark.read().format("textlines")
        .option("maxPartitionBytes", "64k")
        .option("maxBatchBytes", "16k")
        .option("maxBatchRows", "1000")
        .option("includeBytes", String.valueOf(includeBytes));
  }

  private Dataset<Row> read(String path, boolean includeBytes) {
    return reader(includeBytes).load(path);
  }

  @Test
  public void readsTheSameLinesAsSparkText() {
    Dataset<Row> lines = read(dir.getPath(), false);
    assertTrue(lines.rdd().getNumPartitions() > files.size());
    Dataset<Row> expected = spark.read().text(dir.getPath());
    // count() reads no columns
    assertEquals(expected.count(), lines.count());
    assertEquals(0, expected.exceptAll(lines.select("line")).count());
    assertEquals(0, lines.select("line").exceptAll(expected).count());
  }

  @Test
  public void packsSmallFilesIntoPartitions() {
    // Explicit maxPartitionBytes packs all three files into one partition.
    Dataset<Row> packed = spark.read().format("textlines")
        .option("maxPartitionBytes", "128m")
        .option("maxBatchBytes", "16k")
        .load(dir.getPath());
    assertEquals(1, packed.rdd().getNumPartitions());
    Dataset<Row> expected = spark.read().text(dir.getPath());
    assertEquals(0, expected.exceptAll(packed.select("line")).count());
    assertEquals(0, packed.select("line").exceptAll(expected).count());

    // Without maxPartitionBytes, the default scales by defaultParallelism (2 in local[2]).
    Dataset<Row> scaled = spark.read().format("textlines")
        .option("maxBatchBytes", "16k")
        .load(dir.getPath());
    assertEquals(2, scaled.rdd().getNumPartitions());
    assertEquals(expected.count(), scaled.count());
  }

  @Test
  public void doesNotFragmentFileBelowMaxPartitionBytes() {
    File file = files.get(0);
    Map<String, String> opts = new HashMap<>();
    opts.put("path", file.getPath());
    opts.put("maxPartitionBytes", "1m");
    // Smaller than the ~451 KiB file: default FileInputFormat would split at 64 KiB.
    opts.put("fs.local.block.size", "65536");
    TextLinesScan scan = new TextLinesScan(
        new CaseInsensitiveStringMap(opts),
        TextLinesSource.schema(CaseInsensitiveStringMap.empty()));
    InputPartition[] parts = scan.planInputPartitions();
    assertEquals(1, parts.length);
    TextSplitPartition part = (TextSplitPartition) parts[0];
    assertEquals(1, part.paths.length);
    assertEquals(file.length(), part.lengths[0]);
  }

  @Test
  public void usesHadoopSettingsOfTheSessionAndTheRead() throws IOException {
    // Driver side: a minimum split size larger than a file keeps each file in one split.
    String minSplit = "mapreduce.input.fileinputformat.split.minsize";
    String oneGiB = String.valueOf(1L << 30);
    assertTrue(read(dir.getPath(), false).rdd().getNumPartitions() > files.size());
    assertEquals(files.size(),
        reader(false).option(minSplit, oneGiB).load(dir.getPath()).rdd().getNumPartitions());
    spark.conf().set(minSplit, oneGiB);
    try {
      assertEquals(files.size(), read(dir.getPath(), false).rdd().getNumPartitions());
    } finally {
      spark.conf().unset(minSplit);
    }

    // Executor side: LineRecordReader skips lines at or above line.maxlength.
    String maxLen = "mapreduce.input.linerecordreader.line.maxlength";
    long fullCount = spark.read().text(dir.getPath()).count();
    long withoutLongLines = fullCount - (long) LONG_LINES_PER_FILE * files.size();
    assertEquals(withoutLongLines,
        reader(false).option(maxLen, "1000").load(dir.getPath()).count());
    spark.conf().set(maxLen, "1000");
    try {
      assertEquals(withoutLongLines, read(dir.getPath(), false).count());
    } finally {
      spark.conf().unset(maxLen);
    }

    // Executor side: TextInputFormat applies textinputformat.record.delimiter.
    File custom = Files.createTempFile("textlines-delim", ".txt").toFile();
    try {
      Files.write(custom.toPath(), "alpha|beta|gamma".getBytes(StandardCharsets.UTF_8));
      List<Row> rows = spark.read().format("textlines")
          .option("textinputformat.record.delimiter", "|")
          .load(custom.getPath())
          .orderBy("offset")
          .select("line")
          .collectAsList();
      assertEquals(3, rows.size());
      assertEquals("alpha", rows.get(0).getString(0));
      assertEquals("beta", rows.get(1).getString(0));
      assertEquals("gamma", rows.get(2).getString(0));
    } finally {
      custom.delete();
    }
  }

  @Test
  public void offsetsAndBytesMatchTheFile() throws IOException {
    File file = files.get(0);
    byte[] content = Files.readAllBytes(file.toPath());
    List<Row> rows = read(file.getPath(), true).orderBy("offset").collectAsList();
    long next = 0;
    for (Row row : rows) {
      long offset = row.getLong(0);
      byte[] line = row.getString(1).getBytes(StandardCharsets.UTF_8);
      assertEquals(next, offset);
      assertArrayEquals(line, (byte[]) row.get(2));
      byte[] inFile = new byte[line.length];
      System.arraycopy(content, (int) offset, inFile, 0, line.length);
      assertArrayEquals(line, inFile);
      // The next line starts after the terminator
      next = offset + line.length;
      if (next < content.length && content[(int) next] == '\r') {
        next++;
      }
      next++;
    }
    assertEquals(content.length + 1, next);

    // Reordered pruned projection
    List<Row> reordered = read(file.getPath(), true)
        .select("bytes", "offset")
        .orderBy("offset")
        .collectAsList();
    assertEquals(rows.size(), reordered.size());
    for (int i = 0; i < rows.size(); i++) {
      assertEquals(rows.get(i).getLong(0), reordered.get(i).getLong(1));
      assertArrayEquals((byte[]) rows.get(i).get(2), (byte[]) reordered.get(i).get(0));
    }
  }

  @Test
  public void handlesLongLineMultiplePathsAndEmptyDirectory() throws IOException {
    // Single line (5000 B) exceeding maxBatchBytes (1 KiB)
    File first = files.get(0);
    Dataset<Row> longLines = spark.read().format("textlines")
        .option("maxBatchBytes", "1k")
        .load(first.getPath());
    Dataset<Row> firstExpected = spark.read().text(first.getPath());
    assertEquals(0, firstExpected.exceptAll(longLines.select("line")).count());
    assertEquals(0, longLines.select("line").exceptAll(firstExpected).count());

    // Multiple paths via load(p1, p2) JSON array
    Dataset<Row> twoFiles = reader(false).load(files.get(0).getPath(), files.get(1).getPath());
    Dataset<Row> twoExpected = spark.read().text(files.get(0).getPath(), files.get(1).getPath());
    assertEquals(twoExpected.count(), twoFiles.count());
    assertEquals(0, twoExpected.exceptAll(twoFiles.select("line")).count());

    // Both path option and paths JSON array combined
    String pathsJson = "[\"" + files.get(1).getPath().replace("\\", "\\\\")
        + "\",\"" + files.get(2).getPath().replace("\\", "\\\\") + "\"]";
    Dataset<Row> combined = reader(false)
        .option("path", files.get(0).getPath())
        .option("paths", pathsJson)
        .load();
    assertEquals(spark.read().text(dir.getPath()).count(), combined.count());

    // Empty directory stays columnar and returns 0 rows
    File emptyDir = Files.createTempDirectory("textlines-empty").toFile();
    try {
      Dataset<Row> empty = reader(false).load(emptyDir.getPath());
      assertEquals(0, empty.count());
      assertTrue(empty.queryExecution().executedPlan().toString().contains("ColumnarToRow"));
    } finally {
      emptyDir.delete();
    }
  }

  @Test
  public void keepsArrowBuffersWithinTwoMiBWithDefaultLimits() throws IOException {
    int rows = TextLinesScan.DEFAULT_MAX_BATCH_ROWS + 10;
    byte[] emptyLines = new byte[rows];
    for (int i = 0; i < rows; i++) {
      emptyLines[i] = '\n';
    }
    File manyShort = Files.createTempFile("textlines-short", ".txt").toFile();
    try {
      Files.write(manyShort.toPath(), emptyLines);
      FileSplit split = new FileSplit(new Path(manyShort.getPath()), 0, manyShort.length(),
          new String[0]);
      TextSplitPartition partition = TextSplitPartition.of(Collections.singletonList(split));
      Map<String, String> opts = Collections.singletonMap("includeBytes", "true");
      StructType fullSchema = TextLinesSource.schema(new CaseInsensitiveStringMap(opts));
      long twoMiB = 2L << 20;
      try (TextLinesColumnarReader r = new TextLinesColumnarReader(
          new Configuration(), partition, fullSchema,
          TextLinesScan.DEFAULT_MAX_BATCH_ROWS, twoMiB)) {
        assertTrue(r.next());
        ColumnarBatch firstBatch = r.get();
        assertEquals(TextLinesScan.DEFAULT_MAX_BATCH_ROWS, firstBatch.numRows());
        for (int c = 0; c < firstBatch.numCols(); c++) {
          ArrowColumnVector col = (ArrowColumnVector) firstBatch.column(c);
          for (ArrowBuf buf : col.getValueVector().getBuffers(false)) {
            assertTrue("Buffer for col " + c + " exceeded 2 MiB: "
                + buf.getReferenceManager().getSize(),
                buf.getReferenceManager().getSize() <= twoMiB);
          }
        }
        assertTrue(r.next());
        assertEquals(10, r.get().numRows());
        assertFalse(r.next());
      }
    } finally {
      manyShort.delete();
    }
  }

  @Test
  public void validatesOptionsAndScanContracts() {
    StructType schema = TextLinesSource.schema(CaseInsensitiveStringMap.empty());
    String[][] invalid = {
        {"maxBatchRows", "0"},
        {"maxBatchRows", "-1"},
        {"maxBatchBytes", "0"},
        {"maxBatchBytes", "3g"},
        {"maxPartitionBytes", "0"}
    };
    for (String[] pair : invalid) {
      Map<String, String> opts = new HashMap<>();
      opts.put("path", files.get(0).getPath());
      opts.put(pair[0], pair[1]);
      try {
        new TextLinesScan(new CaseInsensitiveStringMap(opts), schema);
        fail("Expected IllegalArgumentException for " + pair[0] + "=" + pair[1]);
      } catch (IllegalArgumentException expected) {
        // expected
      }
    }

    Map<String, String> opts = Collections.singletonMap("path", files.get(0).getPath());
    TextLinesScan s1 = new TextLinesScan(new CaseInsensitiveStringMap(opts), schema);
    TextLinesScan s2 = new TextLinesScan(new CaseInsensitiveStringMap(opts), schema);
    TextLinesScan s3 = new TextLinesScan(new CaseInsensitiveStringMap(opts),
        new StructType().add(TextLinesSource.LINE, schema.fields()[1].dataType(), false));
    assertEquals(Scan.ColumnarSupportMode.SUPPORTED, s1.columnarSupportMode());
    assertEquals(s1, s2);
    assertEquals(s1.hashCode(), s2.hashCode());
    assertNotEquals(s1, s3);

    // Case-insensitive option merge in TextLinesTable: scan option overrides table option.
    Map<String, String> tableOpts = new HashMap<>();
    tableOpts.put("path", files.get(0).getPath());
    tableOpts.put("maxBatchRows", "100");
    TextLinesTable table = new TextLinesTable(schema, new CaseInsensitiveStringMap(tableOpts));
    Map<String, String> scanOpts = Collections.singletonMap("maxbatchrows", "0");
    try {
      table.newScanBuilder(new CaseInsensitiveStringMap(scanOpts)).build();
      fail("Expected scan option maxbatchrows=0 to override table option maxBatchRows=100");
    } catch (IllegalArgumentException expected) {
      // expected
    }
  }
}
