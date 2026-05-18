# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from m3docrag.datasets.m3_docvqa.dataset import M3DocVQADataset
from m3docrag.retrieval.faiss_utils import flatten_page_embeddings, maxsim_top_k_pages
from m3docrag.utils.args import parse_args


def main():
    args = parse_args()

    logger.info("Loading M3DocVQA")

    dataset = M3DocVQADataset(args)

    logger.info(f"Loading M3DocVQA -- all {args.retrieval_model_type} embeddings")

    docid2lens = None
    if args.retrieval_model_type == "colpali":
        docid2embs = dataset.load_all_embeddings()
    elif args.retrieval_model_type == "colbert":
        docid2embs, docid2lens = dataset.load_all_embeddings()

    # len(docid2embs)
    # docid2embs_page_reduced = reduce_embeddings(docid2embs, dim='page')
    # docid2embs_token_reduced = reduce_embeddings(docid2embs, dim='token')
    # docid2embs_page_token_reduced = reduce_embeddings(docid2embs, dim='page_token')

    # flat_doc_embs = []
    # for doc_id, doc_emb in docid2embs.items():
    #     flat_doc_embs += [doc_emb]

    # flat_doc_embs  = torch.cat(flat_doc_embs, dim=0)

    # logger.info(flat_doc_embs.shape)

    d = 128
    quantizer = faiss.IndexFlatIP(d)

    if args.faiss_index_type == "flatip":
        index = quantizer

    elif args.faiss_index_type == "ivfflat":
        ncentroids = 1024
        index = faiss.IndexIVFFlat(quantizer, d, ncentroids)
    else:
        nlist = 100
        m = 8
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    logger.info("Flattening all PDF pages")

    all_token_embeddings, token2pageuid = flatten_page_embeddings(
        docid2embs=docid2embs,
        dim=d,
        docid2lens=docid2lens,
    )
    logger.info(all_token_embeddings.shape)
    logger.info(len(token2pageuid))

    logger.info("Creating index")

    index.train(all_token_embeddings)
    index.add(all_token_embeddings)

    Path(args.output_dir).mkdir(exist_ok=True)
    index_output_path = str(Path(args.output_dir) / "index.bin")
    logger.info(f"Saving index at {index_output_path}")

    faiss.write_index(index, index_output_path)

    logger.info("Running an example query")

    # Example query (should be np.float32)
    example_text_query_emb = np.random.randn(20, 128).astype(np.float32)

    # NN search
    k = 10
    _, nn_indices = index.search(example_text_query_emb, k)

    top_k_pages = maxsim_top_k_pages(
        query_emb=example_text_query_emb,
        nn_indices=nn_indices,
        token2pageuid=token2pageuid,
        all_token_embeddings=all_token_embeddings,
        k=k,
    )

    logger.info("Top-k page candidates with scores:")
    for page_uid, score in top_k_pages:
        logger.info(f"{page_uid} with score {score}")


if __name__ == "__main__":
    main()
