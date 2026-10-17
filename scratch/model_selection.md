# Model selection

## Comparison

| Model | Mean recall@5 | Energy kWh / 1,000 images | Error |
| --- | --- | --- | --- |
| vit_b_32_openai | 0.982000 | 0.005182 |  |
| vit_b_32_laion2b | 0.989000 | 0.002501 |  |
| vit_b_16_siglip_webli | 0.984000 | 0.006101 |  |
| convnext_base_w_laion2b | 0.992000 | 0.006847 |  |

## Run conditions

- Device: cpu
- Dataset: nlphuji/flickr_1k_test_image_text_retrieval (images: 100)
- Energy method: estimated_from_runtime (PUE: 1.0)
- Energy is estimated from wall-clock runtime, not read from a power sensor; treat it as a runtime proxy.

## Selection rule

Choose the model with the highest mean recall@5. Models within 0.01 (one percentage point) of the best recall are tied; among those, choose the lowest energy per 1,000 images.

**Winner:** vit_b_32_laion2b
**Runner-up:** vit_b_32_openai
