# Model selection

## Comparison

| Model | Mean recall@5 | Energy kWh / 1,000 images | Error |
| --- | --- | --- | --- |
| vit_b_32_openai | 0.892000 | 0.002582 |  |
| vit_b_32_laion2b | 0.922700 | 0.002358 |  |
| vit_b_16_siglip_webli | 0.951500 | 0.004983 |  |
| convnext_base_w_laion2b | 0.947500 | 0.006354 |  |

## Run conditions

- Device: cpu
- Dataset: nlphuji/flickr_1k_test_image_text_retrieval (images: 1000)
- Energy method: estimated_from_runtime (PUE: 1.0)
- Energy is estimated from wall-clock runtime, not read from a power sensor; treat it as a runtime proxy.

## Selection rule

Choose the model with the highest mean recall@5. Models within 0.01 (one percentage point) of the best recall are tied; among those, choose the lowest energy per 1,000 images.

**Winner:** vit_b_16_siglip_webli
**Runner-up:** convnext_base_w_laion2b
