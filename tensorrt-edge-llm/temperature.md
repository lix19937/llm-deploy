Temperature（温度） 是一个调节生成文本随机性和创造性的关键超参数。 它本质上是一个数学缩放因子，作用于模型输出层的 Logits（未归一化的概率得分）

```cpp

std::vector<float> softmaxRef(std::vector<float> const& logits, float temperature)
{
    std::vector<float> scaledLogits(logits.size());
    float invTemp = (temperature < 1e-3f) ? 1000.0f : 1.0f / temperature;  
    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaledLogits[i] = logits[i] * invTemp; ////++++++++++++++++++++++++++++++
    }

    // 1th  Find max for numerical stability
    float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());

    // 2th calc      M & d  
    std::vector<float> probs(logits.size());
    float sumExp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] = std::exp(scaledLogits[i] - maxLogit);
        sumExp += probs[i];
    }

    // 3th  
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] /= sumExp;
    }

    return probs;
}
```
