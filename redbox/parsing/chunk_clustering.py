from functools import reduce
from itertools import compress
from typing import Optional

import numpy as np
import scipy
from langchain_core.embeddings.embeddings import Embeddings

from redbox.models.file import Chunk, Metadata


def cluster_chunks(
    chunks: list[Chunk],
    embedding_model: Embeddings,
    desired_chunk_size: int = 300,
    dist_weight_split: float = 0.2,
    dist_use_log: bool = True,
) -> list[Chunk]:
    """Merge together adjacent chanks based on their semantic similarity and token count.

    Args:
        chunks (list[Chunk]): List of raw (small) chunks extracted from document.
        desired_chunk_size (int): Avarage size of the output chunks. Defaults to 300,
        embedding_model (Embeddings): the sentence embedding model used to compare chunk similarity
        dist_weight_split (float): Expects value between 0 and 1.
            When calculating the combined distance metric this is the relative weight (importance)
            of the semantic similarity vs the token counts. Defaults to .2.
        dist_use_log (bool): When calculating the combined distance metric should the input values
            be scaled by log. Defaults to True.

    Returns:
        list[Chunk]: A list of all the (merged) chunks extracted from the given file.
    """
    # filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.token_count > 0]  # type: ignore
    if len(chunks) < 2:
        out_chunks = chunks
    else:
        token_counts = [chunk.token_count for chunk in chunks]  # type: ignore
        # calculate simple vector embedding and distances between adjacent chunks

        chunk_embedding = embedding_model.embed_documents([chunk.text for chunk in chunks])

        pair_embed_dist = [0] + [
            scipy.spatial.distance.cosine(chunk_embedding[i], chunk_embedding[i + 1]) for i in range(len(chunks) - 1)
        ]
        # create distance vector (upper triangular) by combining the token counts with embedding distance
        dist_triu = create_pdist(
            token_counts=token_counts,
            pair_embed_dist=pair_embed_dist,
            weight_embed_dist=dist_weight_split,
            use_log=dist_use_log,
        )

        # cluster the small chunks and cut tree based on desired chunk size
        # Distance approach is Farthest Point Algorithm (complete linkage) which
        # gets the maximum distance between all the points in the cluster
        hc = scipy.cluster.hierarchy.linkage(dist_triu, "complete")
        num_clusters = round(np.sum(token_counts) / desired_chunk_size)  # type: ignore
        out_clusters = [lab[0] for lab in scipy.cluster.hierarchy.cut_tree(hc, n_clusters=num_clusters)]
        # merge clusters and create output chunks
        out_chunks = []
        for i, clust in enumerate(np.unique(out_clusters)):
            chunks_in = list(compress(chunks, out_clusters == clust))
            # if there is only one chunk in the cluster, just use it
            if len(chunks_in) == 1:
                new_chunk = Chunk(**{k: v for k, v in chunks_in[0].model_dump().items() if k != "index"}, index=i)
            else:
                # if there are multiple chunks in the cluster, merge them
                new_chunk = Chunk(
                    parent_file_uuid=chunks_in[0].parent_file_uuid,
                    index=i,
                    text=" ".join([chunk.text for chunk in chunks_in]),
                    metadata=merge_chunk_metadata([chunk.metadata for chunk in chunks_in]),
                    creator_user_uuid=chunks_in[0].creator_user_uuid,
                )
            out_chunks.append(new_chunk)
    return out_chunks


def create_pdist(token_counts, pair_embed_dist, weight_embed_dist=0.2, use_log=True):
    """
    Creates a distance (upper) matrix for the chunk merging.
    It combines embedding distance with token counts metric for adjacent chunks.
    Distance between neighbours is always smaller than further away pair -> enforcing
    the hierarchical clustering to merge only adjacent blocks in each step.

    """
    n = len(token_counts)

    # Phase 1: Calculate the two forms of distance between adjacent chunks

    # embedding distance between chunk i and j is taken as MAXIMUM of the pairwise embedding
    # distance of all the adjacent pairs between them
    embed_dims = np.tri(n, k=0) * np.array(pair_embed_dist)

    # Chebyshev distance is used to make sure that the distance between i and j is always
    # smaller than the distance between i and k and j and k for any k
    embed_dist = scipy.spatial.distance.pdist(embed_dims, "chebyshev")

    # the token count distance between junk and i and j is the size of minimal text segment
    # containing them, i.e. sum of token counts of all the intermediate chunks
    token_dims = np.tri(n + 1, k=0) * np.concatenate([[0], np.array(token_counts)])

    # drop diagonal (sizes of individual chunks)
    drop_ind = [y - x > 1 for x, y in zip(np.triu_indices(n + 1, k=1)[0], np.triu_indices(n + 1, k=1)[1], strict=False)]

    # calculate the token count distance between chunk i and j
    token_dist = scipy.spatial.distance.pdist(token_dims, "cityblock")[drop_ind]

    # scale the distances by log to make them more comparable
    if use_log:
        embed_dist = np.log(embed_dist + 1)
        token_dist = np.log(token_dist + 1)

    # make the two distances comparable and then scale them using input weight parameter
    # smaller weight means more importance of the token count distance
    # bigger weight means more importance of the embedding distance
    if np.std(embed_dist) > 0:
        embed_dist = embed_dist / np.std(embed_dist) * weight_embed_dist
    if np.std(embed_dist) > 0:
        token_dist = token_dist / np.std(token_dist) * (1 - weight_embed_dist)

    # Phase 2: Combine the two distances into one

    # the two above distance are combined either using sum or product (i.e. use_log=T)
    combined_dist = [x + y for x, y in zip(embed_dist, token_dist, strict=False)]
    return combined_dist


def merge_chunk_metadata(meta_in: list[Optional[Metadata]]) -> Optional[Metadata]:
    """
    Combine metadata for multiple chunks from the same document.
    """
    return reduce(Metadata.merge, meta_in)
