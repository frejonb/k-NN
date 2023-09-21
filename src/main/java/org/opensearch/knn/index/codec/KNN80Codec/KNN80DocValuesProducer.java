/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import com.google.common.collect.ImmutableMap;
import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.*;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.opensearch.common.io.PathUtils;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNIndexShard;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.concurrent.ExecutionException;

import static org.apache.lucene.codecs.CodecUtil.FOOTER_MAGIC;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.IndexUtil.getParametersAtLoading;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import org.opensearch.knn.index.memory.NativeMemoryAllocation;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import static org.opensearch.knn.plugin.stats.KNNCounter.GRAPH_QUERY_ERRORS;

/**
 * This class reads the KNN docvalues from the segments
 */
@Log4j2
class KNN80DocValuesProducer extends DocValuesProducer implements Closeable {

    private final Logger logger = LogManager.getLogger(KNN80DocValuesProducer.class);

    private final DocValuesProducer delegatee;
    private final SegmentReadState state;

    private final NativeMemoryCacheManager nativeMemoryCacheManager;

    KNN80DocValuesProducer(DocValuesProducer delegatee, SegmentReadState state) {
        this.delegatee = delegatee;
        this.state = state;
        this.nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();


        for (FieldInfo field : this.state.fieldInfos) {
            if (isKNNBinaryFieldRequired(field)) {
                loadGraphIntoCache(field);
            }
        }
    }
    private void loadGraphIntoCache(FieldInfo field) {

        String directory = ((FSDirectory) FilterDirectory.unwrap(state.segmentInfo.dir)).getDirectory().toString();

        String engineName = field.attributes().getOrDefault(KNN_ENGINE, KNNEngine.NMSLIB.getName());
        KNNEngine knnEngine = KNNEngine.getEngine(engineName);
        String spaceTypeName = field.attributes().getOrDefault(SPACE_TYPE, SpaceType.L2.getValue());
        SpaceType spaceType = SpaceType.getSpace(spaceTypeName);

        /*
         * In case of compound file, extension would be <engine-extension> + c otherwise <engine-extension>
         */
        String engineExtension = state.segmentInfo.getUseCompoundFile()
                ? knnEngine.getExtension() + KNNConstants.COMPOUND_EXTENSION
                : knnEngine.getExtension();
        String engineSuffix = field.name + engineExtension;
        List<String> engineFiles=state.segmentInfo.files().stream().filter(fileName -> fileName.endsWith(engineSuffix)).collect(Collectors.toList());

        if (engineFiles.isEmpty()) {
            log.debug("[KNN] No engine index found for field {} for segment {}", field.name, state.segmentInfo.name);
            return ;
        }

        Path indexPath = PathUtils.get(directory, engineFiles.get(0));
        KNNCounter.GRAPH_QUERY_REQUESTS.increment();


        // We need to first get index allocation
        String indexName = "footest";
        try {
            nativeMemoryCacheManager.get(
                    new NativeMemoryEntryContext.IndexEntryContext(
                            indexPath.toString(),
                            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                            getParametersAtLoading(spaceType, knnEngine, indexName),
                            indexName
                    ),
                    true
            );
        } catch (ExecutionException e) {
            GRAPH_QUERY_ERRORS.increment();
            throw new RuntimeException(e);
        }
    }
    private boolean isKNNBinaryFieldRequired(FieldInfo field) {
        final KNNEngine knnEngine = getKNNEngine(field);
        log.debug(String.format("Read engine [%s] for field [%s]", knnEngine.getName(), field.getName()));
        return field.attributes().containsKey(KNNVectorFieldMapper.KNN_FIELD)
                && KNNEngine.getEnginesThatCreateCustomSegmentFiles().stream().anyMatch(engine -> engine == knnEngine);
    }

    private KNNEngine getKNNEngine(@NonNull FieldInfo field) {
        final String modelId = field.attributes().get(MODEL_ID);
        if (modelId != null) {
            var model = ModelCache.getInstance().get(modelId);
            return model.getModelMetadata().getKnnEngine();
        }
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }
    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        return delegatee.getNumeric(field);
    }


    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        return delegatee.getBinary(field);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        return delegatee.getSorted(field);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        return delegatee.getSortedNumeric(field);
    }

    @Override
    public  SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        return delegatee.getSortedSet(field);
    }

    @Override
    public  void checkIntegrity() throws IOException {
        delegatee.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        delegatee.close();
    }

    @Override
    public DocValuesProducer getMergeInstance() {
        return delegatee.getMergeInstance();
    }
}
