# Overview 
This repository implements a zero-shot video retrieval pipeline that takes in a natural language query and returns the most relevant timestamped clips from a video. The main goal is to retrieve events that may depend on several types of information including: 
- Visual appearance and actions
- Spoken dialogue
- OCR / visible text
- Video metadata and scene descriptions
- Temporal context across multiple time scales

## Approach
The pipeline follows a coarse to fine retrieval strategy:

Video

↓

Hierarchical Video Chunking into Coarse-to-Fine Lengths

↓

Feature / Metadata Extraction from Video Chunks

↓

Multimodal Embeddings + Search Indexes

↓

Natural-Language Query

↓

Query Planner

↓

Video / Transcript / Metadata Retrieval

↓

Evidence / Potential Candidate Aggregation

↓

Candidate Ranking

↓

Recursive Temporal Refinement

↓

Verification

↓

Final Timestamp(s) + Video Clip(s)

Instead of processing every frame of a long video for every query, the video is processed ahead of time and divided into searchable chunks. The system first identifies broad regions that may contain the requested event and then progressively searches smaller intervals to determine more precise timestamps.

## Summary of the different files

- `video_retrieval/config.py` — defines model names, endpoints, and API-key helpers.
- `video_retrieval/video.py` — video chunking, VLM clip caching, frame extraction, and result materialization.
- `video_retrieval/embeddings.py` — Creates Gemini embeddings for videos/text and FAISS indexing/searching.
- `video_retrieval/metadata.py` — VLM metadata generation/indexing/search.
- `video_retrieval/transcript.py` — Whisper transcription, semantic transcript search, and BM25 scores.
- `video_retrieval/retrieval.py` — single-prompt router/query planner, multimodal retrieval, fusion, and candidate clustering.
- `video_retrieval/verification.py` — temporal verification, multi-instance handling, boundary refinement, and temporal NMS.
- `video_retrieval/visual_text.py` — open-vocabulary target-text extraction from arbitrary text-bearing objects/regions.
- `video_retrieval/ocr.py` — compatibility wrappers for older OCR calls; new code normally does not call this directly.
- `video_retrieval/pipeline.py` — top-level `RetrievalResources` and `VideoRetrievalPipeline` API.
- `model_completed.ipynb` - Jupyter notebook to be run for demo 


## Architecture

### 1. Hierarchical Video Chunking

The video is divided into multiple temporal scales, such as:

Large chunks – broad sections of the video
Medium chunks – narrower temporal regions
Small chunks – fine-grained candidate intervals

Videos are divided into in order to allow chunks of different lengths to highlight broader context (coarse) to specific actions (fine) and allow retrieval to move from a broad matching section to increasingly precise timestamps.

### 2. Video Embeddings

Video chunks are converted into numerical embeddings so that they can be compared against text queries through a shared embedding space. This allows text queries to be compared against video chunks. A FAISS index is then constructed over the embeddings for fast similarity search.

### 3. Transcript Retrieval

Audio is transcribed and associated with absolute timestamps. Transcript information is split into searchable segments so queries based on speech can be retrieved independently of visual information, helping to resolve queries related to audio.
The system can use both semantic embedding similarity and BM25 keyword search, as BM25 is useful for exact words, names, or phrases, while embeddings are better for semantic similarity.

### 4. Visual Metadata

Video chunks can also be passed through a vision-language model to generate text descriptions of what occurs in the scene in a specific JSON format.

Example metadata might look like: A police officer approaches a stopped vehicle at night. The driver is visible through the window.

Metadata retrieval provides another way of searching visual information without relying entirely on the raw video embedding, improving redundancy.

### 5. OCR / Visual Text

Frames can also be processed to extract visible text. OCR results are timestamped and added to the searchable evidence associated with each video interval. This helps to resolve queries that may focus on extracting text from a video. 

### 6. Query Planning

Different questions require different retrieval channels, so the query planner converts the original query into specialized searches, such as: 

{
    "visual_queries": [...],
    "transcript_queries": [...],
    "metadata_queries": [...]
}

This allows different retrievers to focus on the part of the query they are best suited to answer. The query is also decomposed in order to allow for more targeted and accurate searching. 

### 7. Evidence Aggregation

Each retrieval system produces evidence for particular time intervals (candidates). These results are combined into a temporal evidence map, and intervals supported by multiple independent retrieval channels receive stronger evidence for being a valid candidate. This makes retrieval more robust than simply returning the highest FAISS similarity score through considereding multiple modalities.

### 8. Candidate Ranking

Candidate intervals are ranked using signals such as embedding similarity, transcript similarity, metadata similarity, BM25 score, and query relevance. Scores from different retrieval systems are calibrated before they are combined because their raw values are not necessarily directly comparable, resulting in a ranked collection of candidate intervals.

### 9. Recursive Temporal Refinement

Once a promising broad interval is identified, the system searches its child/sub chunks. This coarse-to-fine strategy greatly reduces the amount of video that must be examined during the final stages of retrieval.

### 10. Verification

The highest-ranking candidate clips are inspected again using a VLM using visual, transcript, or multimodal evidence, which identifies false positives and helps reject clips that were semantically similar but did not actually contain the requested event.

### 11. Final Clip Extraction

Once the final timestamps have been determined, FFmpeg extracts the matching portion directly from the original video.

## Running the Code 

The easiest way to run the project is through the included Jupyter Notebook.

### 1. Install Dependencies

Install the Python packages required by the project by running pip install -r requirements.txt

FFmpeg is also required for video processing and final clip extraction.

On macOS:

brew install ffmpeg

### 2. Configure the Video

Set the path to the video that should be indexed.

For example: video_path = "videos/example.mp4"

### 3. Run cells in the notebook in order 

### 4. Run queries 

New queries can be entered by modifying the PROMPT variable. 

## Architecture Decisions 

1. Hierarchical chunking: I used multi-scale video chunks because large chunks preserve the context needed to understand an event while smaller chunks provide more accurate timestamps, giving the system both efficient search and precise localization. 
2. Separate retrieval channels: I kept video, transcript, metadata, and OCR retrieval separate because each modality captures different information, allowing the system to route a query toward the strongest source instead of forcing every query through one model. 
3. FAISS semantic search: I used FAISS because it can search large collections of embedding vectors much faster than directly comparing a text query against every stored video chunk.
4. Semantic search + BM25: I combined embedding-based semantic search with BM25 because embeddings are better at finding conceptually similar content while BM25 is more reliable for exact words, names, or phrases that may otherwise be missed.
5. Evidence-based ranking: I combined evidence from multiple retrieval methods because agreement between video, transcript, and metadata signals is generally more reliable than selecting a result based on one similarity score alone.
6. Zero-shot pretrained models: I relied on pretrained foundation models because this avoids the need for a labeled training dataset and allows the same system to handle many different types of videos and natural-language queries.
7. Recursive refinement: I refined high-scoring coarse intervals into smaller child chunks because this avoids performing expensive fine-grained search across the entire video while still producing precise start and end timestamps.
8. Cached preprocessing: I saved embeddings, metadata, transcripts, and search indexes because most of this information does not change between queries, making repeated searches over the same video significantly faster.
9. Query planning: I used a query planner to break a natural-language request into modality-specific searches because different parts of a query may require visual, transcript, OCR, or temporal evidence.
10. Temporal evidence aggregation: I mapped retrieval results back onto a shared video timeline because overlapping evidence from different sources makes it easier to identify the sections where an event is most likely to occur.
11. Final verification: I added a verification stage because retrieval models can return clips that are semantically related but do not actually contain the requested event, so checking the strongest candidates helps reduce false positives.
12. Separate preprocessing and retrieval: I separated the expensive video-processing stage from query-time retrieval so the video only needs to be indexed once, after which many different user queries can be answered efficiently.

## What I Tried
1. I initially tried implementing just single chunks, but it was difficult to determine the optimal length of the chunk that would capture as much information as possible. Thus, I decided to use hierarchical chunking instead. 
2. I had initially tried solely relying on video and text embeddings, but I realized that this would not work well for more general prompts/prompts that required more reasoning, whihc led me to implement more retrieval channels. 
3. I experimented with combining both semantic and BM25 metrics for audio analysis, since I noticed that exact language could behave differently from semantic meaning. 
4. I tried implementing OCR for text-extraction queries specifically to extract text from videos, rather than just returning the video frames. 

## Potential Tradeoffs 
1. Accuracy vs Computation: More temporal scales and more retrieval channels improve the amount of available evidence but require additional storage and preprocessing, which may also be affected by OpenRouter's rate limits. 
2. Context vs. temporal precision: Long clips tend to provide more contextual information, while shorter ones contain more accurate boundaries and specific actions. 
3. Query Latency: I tried implementing precomputation for one video to reduce the amount of latency per query. 
4. Generalized vs Specialized Inference: Since I used pretrained models, I didn't have to annotate any data specifically or retrain any models, which can be costly. However, training more specialized models could yield better performance. 

## Future Works
1. Improve OCR: Build a more robust video OCR pipeline using text detection, frame enhancement, multi-frame aggregation, and tracking to better recognize small or blurry text such as license plates and signs.
2. Add stronger temporal reasoning: Extend the query planner to understand relationships such as before, after, during, and then, through analyzing more video chunks,  allowing the system to handle queries involving sequences of multiple events.
3. Improve ranking and score calibration: Tune or learn how much weight to give video, transcript, metadata, OCR, and BM25 evidence for each query instead of relying mainly on manually selected scoring rules.
4. Build an annotated formal evaluation benchmark to evaluate model performance