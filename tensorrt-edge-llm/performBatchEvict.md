
投机采样（Speculative Decoding）推理框架中的 **Batch 驱逐与cache 压缩**（Batch Eviction & Compaction） 逻辑。     
简单来说，当一个 Batch 中有部分请求（Sequence）已经生成完毕（Finished）时，为了节省显存并提高后续迭代的效率，代码会将这些“已完成”的请求从当前的活跃队列中剔除，并紧凑化（Compact）剩余的显存空间。

1. 状态检测与映射建立    
代码首先检查 context.finishedStates。如果有请求完成，它会调用 buildBatchMapping。     
+ Batch Mapping 的作用：生成一个索引映射表。     
例如，原来的 Batch 索引是 [0, 1, 2, 3]，如果 1 和 2 结束了，映射表可能是 [0, -1, -1, 1]（表示原索引 0 移到新位置 0，原索引 3 移到新位置 1，其余剔除）。    

2. GPU 显存压缩   
这是代码最重头的部分。因为投机采样涉及两个模型（Base 和 Draft），必须同步清理两者的缓存。         
+ KV Cache 压缩：调用 kernel::compactKVCache。这通常是一个高效的 CUDA Kernel，将显存中不连续的、属于“存活请求”的 KV 缓存块重新排列到连续的内存地址。   
+ RoPE 缓存压缩：处理旋转位置编码（RoPE）相关的 Cos/Sin 缓存。    
+ 中间张量压缩：包括 mBaseHiddenStatesOutput（隐藏层状态）、mAcceptedTokenIds（已接受的 Token）等。这些是下一轮投机验证（Accept Token）时必须读取的上下文。

3. 结果保存与 CPU 数据清理       
在正式销毁数据前，代码会将已完成请求的元数据（TokenIds、生成长度、系统提示词等）转移到 context.evictedTokenIds 等容器中。  
+ 使用 std::move 来避免大向量（如生成的长文本）的拷贝开销。     
+ rt::compactVector：对 CPU 端的 std::vector 进行原地压缩，移除无效元素。   

------------------------------------------    
4. 细节   
+ 双模型同步：代码显式地处理了 mBaseEngineRunner 和 mDraftEngineRunner。在投机采样中，如果 Base 模型剔除了一波请求，Draft 模型必须同步剔除，否则两者的 KV Cache 会由于索引错位而导致推理崩坏。      
+ 多模态支持（MRope）：代码中检测 getRopeCosSinCacheTensor 是否为 3 维，这暗示了系统可能支持多模态（如图像+文本），需要对每 Batch 的位置编码进行特殊处理。     
+ 异步与同步平衡：     
使用 cudaMemcpyAsync 将映射表上传到 GPU，尽可能减少 Host 到 Device 的阻塞。    
但在保存 CPU 结果前调用了 cudaStreamSynchronize。这是因为 CPU 需要读取上一轮 GPU 计算出的结果（如 mAcceptLength），必须确保 GPU 已经算完。

-------------------------------------------     
为什么要这样做？     
+ 显存碎片化：如果不压缩，随着请求不断结束，显存中会出现大量空洞。对于自动驾驶（Orin-X）等显存受限的边缘平台，这会导致无法装载新的 Prompt。   
+ 算力对齐：GPU 算子（如矩阵乘法）在 Batch 连续时效率最高。通过 compact 操作，将 10 个零散的请求压成前 10 个连续请求，能更好地填充 Tensor Core。    
+ 投机循环的特殊性：投机采样每轮会输出多个词。如果第 3 个请求在这一轮刚好达到 stop_token，就需要立即在这一轮结束后的 Runtime 里把它“剥离”，否则下一轮 Draft 模型还会浪费算力去为它“猜测”后续词。     


```cpp
bool LLMInferenceSpecDecodeRuntime::performBatchEvict(SpecDecodeInferenceContext& context)
{
    // Check if any batch has finished
    bool hasFinishedBatch = false;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            hasFinishedBatch = true;
            break;
        }
    }

    // no batch finished 
    if (!hasFinishedBatch)
    {
        return true;
    }

    int32_t const oldActiveBatch = context.activeBatchSize;

    // Build batch mapping
    std::vector<int32_t> batchMapping = buildBatchMapping(context.finishedStates);

    // Calculate new active batch size
    int32_t newActiveBatch = 0;
    for (auto newIdx : batchMapping)
    {
        if (newIdx >= 0)
        {
            newActiveBatch = std::max(newActiveBatch, newIdx + 1);
        }
    }

    // Log eviction details
    std::vector<int32_t> evictedIndices;
    for (int32_t i = 0; i < oldActiveBatch; ++i)
    {
        if (batchMapping[i] < 0)
        {
            evictedIndices.push_back(i);
        }
    }
    // TODO: format the log message in a more readable way such as vector print.
    LOG_DEBUG("Batch eviction: %d active batches to %d remaining (evicted %d batch(es): indices [%s])", oldActiveBatch,
        newActiveBatch, static_cast<int32_t>(evictedIndices.size()),
        [&evictedIndices]() {
            std::string result;
            for (size_t i = 0; i < evictedIndices.size(); ++i)
            {
                if (i > 0)
                {
                    result += ", ";
                }
                result += std::to_string(evictedIndices[i]);
            }
            return result;
        }()
            .c_str());

    // Upload batch mapping to GPU
    mDeviceBatchMapping.reshape({oldActiveBatch});
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(), oldActiveBatch * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    // Compact Base KV Cache
    auto& baseLinearKVCache = mBaseEngineRunner->getLinearKVCache();
    rt::Tensor baseKVCacheBuffer = baseLinearKVCache.getKVCacheBuffer();
    kernel::compactKVCache(mDeviceBatchMapping, baseKVCacheBuffer, baseLinearKVCache.getKVCacheLengths(),
        oldActiveBatch, newActiveBatch, context.stream);
    baseLinearKVCache.setActiveBatchSize(newActiveBatch);

    // Compact Draft KV Cache
    auto& draftLinearKVCache = mDraftEngineRunner->getLinearKVCache();
    rt::Tensor draftKVCacheBuffer = draftLinearKVCache.getKVCacheBuffer();
    kernel::compactKVCache(mDeviceBatchMapping, draftKVCacheBuffer, draftLinearKVCache.getKVCacheLengths(),
        oldActiveBatch, newActiveBatch, context.stream);
    draftLinearKVCache.setActiveBatchSize(newActiveBatch);

    // Compact Draft Model's RoPE CosSin Cache if it's per-batch (MRope for multimodal)
    rt::Tensor& draftRopeCache = mDraftEngineRunner->getRopeCosSinCacheTensor();
    if (draftRopeCache.getShape().getNumDims() == 3 && draftRopeCache.getShape()[0] == oldActiveBatch
        && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            draftRopeCache, mDeviceBatchMapping, draftRopeCache, oldActiveBatch, newActiveBatch, context.stream);
        auto const seqLen = static_cast<int32_t>(draftRopeCache.getShape()[1]);
        auto const rotaryDim = static_cast<int32_t>(draftRopeCache.getShape()[2]);
        draftRopeCache.reshape({newActiveBatch, seqLen, rotaryDim});
    }

    // Compact Base Model's RoPE CosSin Cache if it's per-batch (MRope for multimodal)
    rt::Tensor& baseRopeCache = mBaseEngineRunner->getRopeCosSinCacheTensor();
    if (baseRopeCache.getShape().getNumDims() == 3 && baseRopeCache.getShape()[0] == oldActiveBatch
        && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            baseRopeCache, mDeviceBatchMapping, baseRopeCache, oldActiveBatch, newActiveBatch, context.stream);
        auto const seqLen = static_cast<int32_t>(baseRopeCache.getShape()[1]);
        auto const rotaryDim = static_cast<int32_t>(baseRopeCache.getShape()[2]);
        baseRopeCache.reshape({newActiveBatch, seqLen, rotaryDim});
    }

    // Compact cross-round GPU tensors that are read (not just written) in the next round

    // 1. mBaseHiddenStatesOutput: read by runDraftModelAcceptToken in next round
    //    Shape: [activeBatchSize, maxAcceptDepth, baseHiddenDim]
    if (mBaseHiddenStatesOutput.getShape().getNumDims() == 3 && mBaseHiddenStatesOutput.getShape()[0] == oldActiveBatch
        && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(mBaseHiddenStatesOutput, mDeviceBatchMapping, mBaseHiddenStatesOutput,
            oldActiveBatch, newActiveBatch, context.stream);
        auto const dim1 = static_cast<int32_t>(mBaseHiddenStatesOutput.getShape()[1]);
        auto const dim2 = static_cast<int32_t>(mBaseHiddenStatesOutput.getShape()[2]);
        mBaseHiddenStatesOutput.reshape({newActiveBatch, dim1, dim2});
    }

    // 2. mAcceptedTokenIds: read by runDraftModelAcceptToken to prepare input IDs
    //    Shape: [activeBatchSize, maxAcceptDepth]
    if (mAcceptedTokenIds.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            mAcceptedTokenIds, mDeviceBatchMapping, mAcceptedTokenIds, oldActiveBatch, newActiveBatch, context.stream);
        auto const maxAcceptDepth = static_cast<int32_t>(mAcceptedTokenIds.getShape()[1]);
        mAcceptedTokenIds.reshape({newActiveBatch, maxAcceptDepth});
    }

    // 3. mAcceptLength: read by runDraftModelAcceptToken to set per-batch accept counts
    //    Shape: [activeBatchSize]
    if (mAcceptLength.getShape()[0] == oldActiveBatch && newActiveBatch > 0)
    {
        kernel::compactTensorBatch(
            mAcceptLength, mDeviceBatchMapping, mAcceptLength, oldActiveBatch, newActiveBatch, context.stream);
        mAcceptLength.reshape({newActiveBatch});
    }

    // Compact CPU context
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Save evicted batches' results before compacting (using original batch index)
    for (size_t i = 0; i < batchMapping.size(); ++i)
    {
        if (batchMapping[i] < 0 && context.finishedStates[i])
        {
            // This batch is evicted and finished, save its results with original index
            int32_t originalIdx = context.batchIndexMapping[i];
            context.evictedTokenIds[originalIdx] = std::move(context.tokenIds[i]);
            context.evictedGenerateLengths[originalIdx] = context.currentGenerateLengths[i];
            context.evictedActualIterations[originalIdx] = context.actualIterations[i];
            context.evictedSystemPrompts[originalIdx] = std::move(context.systemPrompts[i]);
            context.evictedRawBatchedInputIds[originalIdx] = std::move(context.rawBatchedInputIds[i]);
            context.evictedPromptLengths[originalIdx] = context.promptLengths[i];
        }
    }

    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.actualIterations);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.promptLengths);
    rt::compactVector(batchMapping, context.batchIndexMapping);

    // Update active batch size
    context.activeBatchSize = newActiveBatch;

    return true;
}

```
