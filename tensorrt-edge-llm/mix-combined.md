
```cpp
std::set<int32_t> getCombinedAllowedTokensRef(
    std::vector<float> const& logits, int32_t topK, float topP, float temperature)
{
    // When temperature = 0.0f, we should always pick the highest probability token
    // This matches the behavior in SamplingParams constructor
    if (temperature < 1e-3f)
    {
        int32_t idxMax = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        // Return only the highest probability token
        return std::set<int32_t>{idxMax};
    }

    std::vector<std::pair<float, int32_t>> logitPairs;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i)
    {
        logitPairs.emplace_back(logits[i], i);
    }

    // Sort by logits in descending order
    std::sort(logitPairs.begin(), logitPairs.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

    // +++++++++++++++++ 1 +++++++++++++++++++
    // Apply top-k constraint first
    int32_t kLimit = std::min(topK, static_cast<int32_t>(logits.size()));

    // Extract top-k logits and compute probabilities
    std::vector<float> topKLogits(kLimit);
    for (int32_t i = 0; i < kLimit; ++i)
    {
        topKLogits[i] = logitPairs[i].first;
    }
    auto topKProbs = softmaxRef(topKLogits, temperature);

    // +++++++++++++++++ 2 +++++++++++++++++++
    // Apply top-p constraint to the top-k elements
    float cumsum = 0.0f;
    int32_t cutoff = kLimit - 1;
    for (int32_t i = 0; i < kLimit; ++i)
    {
        cumsum += topKProbs[i];
        if (cumsum >= topP)
        {
            cutoff = i;
            break;
        }
    }

    std::set<int32_t> allowedTokens;
    for (int32_t i = 0; i <= cutoff; ++i)
    {
        allowedTokens.insert(logitPairs[i].second);
    }

    return allowedTokens;
}
```
