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

from typing import Dict, List, Tuple

import numpy as np
import torch


def flatten_page_embeddings(
    docid2embs: Dict[str, torch.Tensor],
    dim: int = 128,
    docid2lens: Dict[str, torch.Tensor] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Flatten per-document page embeddings into a single token matrix.

    For ColPali (`docid2lens=None`), each doc embedding is shaped
    `(n_pages, n_tokens, dim)` and every page contributes `n_tokens` rows.
    For ColBERT-style (`docid2lens` provided), each doc embedding is
    `(total_tokens, dim)` already and `docid2lens[doc_id]` gives per-page
    token counts.

    Returns:
        all_token_embeddings: float32 ndarray of shape (total_tokens, dim).
        token2pageuid: list mapping each flat row to "<doc_id>_page<N>".
    """
    all_token_embeddings: List[torch.Tensor] = []
    token2pageuid: List[str] = []

    if docid2lens is None:
        for doc_id, doc_emb in docid2embs.items():
            for page_id in range(len(doc_emb)):
                page_emb = doc_emb[page_id].view(-1, dim)
                all_token_embeddings.append(page_emb)
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * page_emb.shape[0])
    else:
        for doc_id, doc_emb in docid2embs.items():
            all_token_embeddings.append(doc_emb)
            for page_id, page_len in enumerate(docid2lens[doc_id]):
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * page_len.item())

    flat = torch.cat(all_token_embeddings, dim=0).float().numpy()
    return flat, token2pageuid


def maxsim_top_k_pages(
    query_emb: np.ndarray,
    nn_indices: np.ndarray,
    token2pageuid: List[str],
    all_token_embeddings: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    """Aggregate FAISS per-query-token neighbours into page-level MaxSim scores.

    For each query token, take the max dot-product among returned neighbour
    tokens that map to the same page; then sum those per-token maxes across
    query tokens to form the page's final score. Returns top-k pages sorted
    by score.

    Args:
        query_emb: (n_query_tokens, dim) query token embeddings.
        nn_indices: (n_query_tokens, n_neighbours) FAISS index ids.
        token2pageuid: maps flat token id -> "<doc_id>_pageN".
        all_token_embeddings: (total_tokens, dim) the indexed corpus.
        k: number of pages to return.
    """
    final_page2scores: Dict[str, float] = {}

    for q_idx, q_tok in enumerate(query_emb):
        current_page2scores: Dict[str, float] = {}
        for nn_idx in range(nn_indices.shape[1]):
            token_idx = nn_indices[q_idx, nn_idx]
            page_uid = token2pageuid[token_idx]
            score = float((q_tok * all_token_embeddings[token_idx]).sum())

            prev = current_page2scores.get(page_uid)
            current_page2scores[page_uid] = score if prev is None else max(prev, score)

        for page_uid, score in current_page2scores.items():
            final_page2scores[page_uid] = final_page2scores.get(page_uid, 0.0) + score

    sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pages[:k]
