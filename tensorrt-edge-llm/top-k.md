Top-K：强制只看概率最高的前 $K$ 个词（数量固定）， 而Top-P： 根据概率质量决定看多少个词（数量不固定）      

最佳实践通常是先经过 Top-K 过滤掉极低概率词，再通过 Top-P 进行精细核采样，以达到速度与质量的平衡。

```cpp
std::set<int32_t> getTopKAllowedTokensRef(std::vector<float> const& logits, int32_t topK)
{
    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // 基于 logits 降序排序  
    // Sort by logits in descending order
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size())); // 取最小值   
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + kLimit, logitPairs.end(),
        [](auto const& a, auto const& b) { return a.first > b.first; });

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i < kLimit; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}
```
