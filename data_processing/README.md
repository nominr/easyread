# data processing
Steps:
- Format all the data into jsonl. Format: {"id": int, "normal_text": string, "easy_text": list of strings, with each segment}
- Write full sentences standalone sentences from each segment (LLM pipeline), and add into jsonl. Format: {"id": int, "normal_text": string, "easy_text": list of strings, with each segment, "easy_read_rewritten": list of strings, where each segment is now a full standalone sentence }
- Detect all the sentences that seem malformed (use of generic words, that indicate poor sentence formation). Step 1: write those to a separate file, manually correct or delete. Step 2: write these corrected sentences back to the full jsonl with all the data
Pipeline:
[seg file] --segmentation.py--> [data1.jsonl] --rewrite_easy_segments.py--> [rewritten_data.jsonl] --post-processing.py--> [segmentation_fixes.jsonl] (where you can manually edit the segments with issues before writing them back to the full jsonl) + [post_processed_segmentation.py]

Input file: any seg file
FINAL OUTPUT FILE: post_processing_segmentation.jsonl
