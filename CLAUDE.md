# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

M3DocRAG: multi-modal RAG over multi-page, multi-document PDF collections. Research code accompanying the M3DocRAG paper (Cho et al., 2024). Pipeline: ColPali visual page embeddings → FAISS index → top-K page retrieval → multi-modal LM (default Qwen2-VL-7B) for VQA.

## Install / Setup

```bash
pip install -e .
# Poppler required for pdf2image: `conda install -y poppler` or `apt-get install poppler-utils`
```

Python >= 3.10. Pinned deps include `flash-attn==2.5.8`, `bitsandbytes==0.43.1`, `numpy==1.26.4`, `colpali-engine==0.3.1`, `accelerate==1.1.0`. CUDA-only (flash-attn, bitsandbytes).

Paths come from `.env` via python-dotenv (`src/m3docrag/utils/paths.py`):
- `LOCAL_DATA_DIR`, `LOCAL_EMBEDDINGS_DIR`, `LOCAL_MODEL_DIR`, `LOCAL_OUTPUT_DIR`

Model checkpoints (cloned under `$LOCAL_MODEL_DIR`): `colpaligemma-3b-pt-448-base`, `colpali-v1.2`, `Qwen2-VL-7B-Instruct`.

## Pipeline Commands

Three stages, each driven by a `fire`-CLI script in `examples/`:

1. **Embed pages** (per-doc visual embeddings via ColPali):
   `accelerate launch --num_processes=1 --mixed_precision=bf16 examples/run_page_embedding.py ...`
2. **Build FAISS index** (e.g., `ivfflat`):
   `python examples/run_indexing_m3docvqa.py ...`
3. **RAG eval** (retrieve top-K, VQA with `--bits=16` or `--bits=4` for low VRAM):
   `python examples/run_rag_m3docvqa.py ...`

See README.md for full flag sets. Dataset prep lives in `m3docvqa/` (separate README).

## Lint

`ruff` configured in `pyproject.toml` (target `py310`, selects `E4,E7,E9,F,I`). No test suite present.

## Architecture

`src/m3docrag/` layout — each subpackage is a swappable component:

- `retrieval/colpali.py` — `ColPaliRetrievalModel` wraps ColPali backbone + LoRA adapter; produces multi-vector page embeddings; MaxSim scoring; FAISS index build/load helpers.
- `vqa/` — one file per VLM family (`qwen2`, `internvl2`, `idefics2`, `idefics3`, `florence2`). Each exposes a uniform load + generate interface. `qwen2.py` is the default.
- `rag/` — `base.py` / `multimodal.py` compose a retrieval model with a VQA model: embed query → FAISS search → fetch page images → prompt VLM. `utils.py` handles page-image fetching/cropping.
- `datasets/m3_docvqa/` — M3DocVQA loader; yields (question, doc_ids, page images). Other benchmarks (MMLongBench-Doc, MP-DocVQA) are referenced in the paper but only M3DocVQA loaders are in-tree.
- `utils/paths.py` — `.env`-driven path resolution used by all entry scripts.

Key data flow: pages are rendered to images (pdf2image/poppler) → ColPali emits per-patch token embeddings per page → flattened/indexed in FAISS → at query time, query tokens MaxSim-scored against page tokens → top-K page images passed to VLM with the question.

Entry scripts use `fire.Fire` so any function arg is a `--flag`. Quantization (`--bits=4`) routes through bitsandbytes in the VQA loader.

## Conventions

- `loguru` for logging.
- `accelerate` for multi-GPU / mixed precision on the embedding stage.
- No tests; changes are validated by re-running the three-stage pipeline on a small split.
