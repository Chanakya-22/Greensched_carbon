# Model selection

## Comparison

| Model | Mean recall@5 | Energy kWh / 1,000 images | Error |
| --- | --- | --- | --- |
| vit_b_32_openai | 0.886400 | 0.002516 |  |
| vit_b_32_laion2b | 0.922700 | 0.002408 |  |
| vit_b_16_siglip_webli | 0.951500 | 0.005004 |  |
| convnext_base_w_laion2b | 0.947500 | 0.005703 |  |

## Selection rule

Choose the model with the highest mean recall@5. Models within 0.01 (one percentage point) of the best recall are tied; among those, choose the lowest energy per 1,000 images.

**Winner:** vit_b_16_siglip_webli
**Runner-up:** convnext_base_w_laion2b
