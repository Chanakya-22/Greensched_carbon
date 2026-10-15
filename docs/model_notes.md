# Model selection notes

The registry focuses on dual-encoder image/text models that independently
encode images and text into a shared embedding space. This structure supports
precomputing vectors and retrieving candidates with vector similarity.

BLIP and BLIP-2 were excluded from this embedding-search comparison because
their common image-text matching and generation workflows use cross-encoder
or generative components rather than only the independent dual-encoder path
needed for scalable embedding retrieval. BLIP-2 is also a large model family,
which would make this benchmark a poor like-for-like comparison with the four
registered embedding encoders. Their omission is specific to this evaluation
goal; it is not a claim that they are unsuitable for image captioning,
visual question answering, or other multimodal tasks.
