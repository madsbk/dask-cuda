import asyncio
from collections import defaultdict
from operator import getitem
from typing import Dict, List, Optional, Set

from toolz import first

import dask
import dask.dataframe
import distributed
from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import shuffle_group
from dask.delayed import delayed
from distributed import wait
from distributed.protocol import nested_deserialize, to_serialize

from . import comms


async def send(eps, rank_to_out_parts_list: Dict[int, List[List[DataFrame]]]):
    """Notice, items send are removed from `rank_to_out_parts_list`"""
    futures = []
    for rank, ep in eps.items():
        out_parts_list = rank_to_out_parts_list.pop(rank, None)
        if out_parts_list is not None:
            futures.append(ep.write([to_serialize(f) for f in out_parts_list]))
    await asyncio.gather(*futures)


async def recv(
    eps, in_nparts: Dict[int, int], out_parts_list: List[List[List[DataFrame]]]
):
    """Notice, received items are appended to `out_parts_list`"""
    futures = []
    for rank, ep in eps.items():
        if rank in in_nparts:
            futures.append(ep.read())
    out_parts_list.extend(nested_deserialize(await asyncio.gather(*futures)))


def sort_in_parts(
    in_parts: List[Dict[int, DataFrame]],
    rank_to_out_part_ids: Dict[int, List[int]],
    ignore_index: bool,
    concat_dfs_of_same_output_partition: bool,
) -> Dict[int, List[List[DataFrame]]]:
    """ Sort the list of grouped dataframes in `in_parts`

    It returns a dict that for each worker-rank specifies the output partitions:
    '''
        for each worker:
            for each output partition:
                list of dataframes that makes of an output partition
    '''
    If `concat_dfs_of_same_output_partition` is True, all the dataframes of an
    output partition are concatenated.

    Parameters
    ----------
    in_parts: list of dict of dataframes
        List of dataframe groups that need to be shuffled.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
    ignore_index: bool
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.
    concat_dfs_of_same_output_partition: bool
        Concatenate all dataframes of the same output partition.

    Returns
    -------
    rank_to_out_parts_list: dict of list of list of DataFrames
        Dict that maps each worker rank to its output partitions.
    """

    out_part_id_to_dataframes = defaultdict(list)  # part_id -> list of dataframes
    for bins in in_parts:
        for k, v in bins.items():
            out_part_id_to_dataframes[k].append(v)
        del bins

    # Create mapping: rank -> list of [list of dataframes]
    rank_to_out_parts_list: Dict[int, List[List[DataFrame]]] = {}
    for rank, part_ids in rank_to_out_part_ids.items():
        rank_to_out_parts_list[rank] = [out_part_id_to_dataframes[i] for i in part_ids]
    del out_part_id_to_dataframes

    # Concatenate all dataframes of the same output partition.
    if concat_dfs_of_same_output_partition:
        for rank in rank_to_out_part_ids.keys():
            for i in range(len(rank_to_out_parts_list[rank])):
                if len(rank_to_out_parts_list[rank][i]) > 1:
                    rank_to_out_parts_list[rank][i] = [
                        _concat(
                            rank_to_out_parts_list[rank][i], ignore_index=ignore_index
                        )
                    ]
    return rank_to_out_parts_list


async def local_shuffle(
    s,
    workers: Set[int],
    in_nparts: Dict[int, int],
    in_parts: List[Dict[int, DataFrame]],
    rank_to_out_part_ids: Dict[int, List[int]],
    ignore_index: bool,
) -> List[DataFrame]:
    """Local shuffle operation of the already grouped/partitioned dataframes

    This function is running on each worker participating in the shuffle.

    Parameters
    ----------
    s: dict
        Worker session state
    workers: set
        Set of ranks of all the participants
    in_nparts: dict
        dict that for each worker rank specifices the
        number of partitions that worker has of the input dataframe.
        If the worker doesn't have any partitions, it is excluded from the dict.
    in_parts: list of dict of dataframes
        List of dataframe groups that need to be shuffled.
    rank_to_out_part_ids: dict
        dict that for each worker rank specifices a list of partition IDs that
        worker should return. If the worker shouldn't return any partitions,
        it is excluded from the dict.
    ignore_index: bool
        Ignore index during shuffle.  If ``True``, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    partitions: list of DataFrames
        List of dataframe-partitions
    """
    myrank = s["rank"]
    eps = s["eps"]
    assert s["rank"] in workers

    rank_to_out_parts_list = sort_in_parts(
        in_parts,
        rank_to_out_part_ids,
        ignore_index,
        concat_dfs_of_same_output_partition=True,
    )

    # Communicate all the dataframe-partitions all-to-all. The result is
    # `out_parts_list` that for each worker and for each output partition
    # contains a list of dataframes received.
    out_parts_list: List[List[List[DataFrame]]] = []
    futures = []
    if myrank in rank_to_out_parts_list:
        futures.append(recv(eps, in_nparts, out_parts_list))
    if myrank in in_nparts:
        futures.append(send(eps, rank_to_out_parts_list))
    await asyncio.gather(*futures)

    # At this point `send()` should have pop'ed all output partitions
    # beside the partitions owned be `myrank`.
    assert len(rank_to_out_parts_list) == 1

    # Concatenate the received dataframes into the final output partitions
    ret = []
    for i in range(len(rank_to_out_part_ids[myrank])):
        dfs = []
        for out_parts in out_parts_list:
            dfs.extend(out_parts[i])
            out_parts[i] = None
        dfs.extend(rank_to_out_parts_list[myrank][i])
        rank_to_out_parts_list[myrank][i] = None
        if len(dfs) > 1:
            ret.append(_concat(dfs, ignore_index=ignore_index))
        else:
            ret.append(dfs[0])
    return ret


def dataframe_shuffle(
    df: DataFrame,
    column_names: List[str],
    npartitions: Optional[int] = None,
    ignore_index: bool = False,
) -> DataFrame:
    """Order divisions of DataFrame so that all values within column(s) align

    This enacts a task-based shuffle using explicit-comms. It requires a full
    dataset read, serialization and shuffle. This is expensive. If possible
    you should avoid shuffles.

    This does not preserve a meaningful index/partitioning scheme. This is not
    deterministic if done in parallel.

    Requires an activate client.

    Parameters
    ----------
    df: dask.dataframe.DataFrame
        Dataframe to shuffle
    column_names: list of strings
        List of column names on which we want to split.
    npartitions: int or None
        The desired number of output partitions. If None, the number of output
        partitions equals `df.npartitions`
    ignore_index: bool
        Ignore index during shuffle.  If True, performance may improve,
        but index values will not be preserved.

    Returns
    -------
    df: dask.dataframe.DataFrame
        Shuffled dataframe

    Developer Notes
    ---------------
    The implementation consist of three steps:
      (a) Extend the dask graph of `df` with a call to `shuffle_group()` for each
          dataframe partition and submit the graph.
      (b) Submit a task for on each worker that shuffle (all-to-all communicate)
          the groups from (a) and return a list of dataframe-partitions.
      (c) Submit a dask graph that extract (using `getitem()`) individual
          dataframe-partitions from (b).
    """
    c = comms.default_comms()

    # As default we preserve number of partitions
    if npartitions is None:
        npartitions = df.npartitions

    # Step (a): partition/group each dataframe-partition
    name = (
        "explicit-comms-shuffle-group-"
        f"{tokenize(df, column_names, npartitions, ignore_index)}"
    )
    dsk = dict(df.__dask_graph__())
    output_keys = []
    for input_key in df.__dask_keys__():
        output_key = (name, input_key[1])
        dsk[output_key] = (
            shuffle_group,
            input_key,
            column_names,
            0,
            npartitions,
            npartitions,
            ignore_index,
            npartitions,
        )
        output_keys.append(output_key)
    df_groups = compute_as_if_collection(type(df), dsk, output_keys, sync=False)
    wait(df_groups)
    for f in df_groups:  # Check for errors
        if f.status == "error":
            f.result()  # raise exception

    # Step (b): find out which workers has what part of `df_groups`,
    #           find the number of output each worker should have,
    #           and submit `local_shuffle()` on each worker.
    key_to_part = dict([(str(part.key), part) for part in df_groups])
    in_parts = defaultdict(list)  # Map worker -> [list of futures]
    for key, workers in c.client.who_has(df_groups).items():
        # Note, if multiple workers have the part, we pick the first worker
        in_parts[first(workers)].append(key_to_part[key])

    # Let's create a dict that specifices the number of partitions each worker has
    in_nparts = {}
    workers = set()  # All ranks that have a partition of `df`
    for rank, worker in enumerate(c.worker_addresses):
        nparts = len(in_parts.get(worker, ()))
        if nparts > 0:
            in_nparts[rank] = nparts
            workers.add(rank)

    # Find the output partitions for each worker
    div = npartitions // len(workers)
    rank_to_out_part_ids = {}  # rank -> [list of partition id]
    for i, rank in enumerate(workers):
        rank_to_out_part_ids[rank] = list(range(div * i, div * (i + 1)))
    for rank, i in zip(workers, range(div * len(workers), npartitions)):
        rank_to_out_part_ids[rank].append(i)

    # Run `local_shuffle()` on each worker
    result_futures = {}
    for rank, worker in enumerate(c.worker_addresses):
        if rank in workers:
            result_futures[rank] = c.submit(
                worker,
                local_shuffle,
                workers,
                in_nparts,
                in_parts[worker],
                rank_to_out_part_ids,
                ignore_index,
            )
    distributed.wait(list(result_futures.values()))

    # Step (c): extract individual dataframe-partitions
    ret = []
    for rank, parts in rank_to_out_part_ids.items():
        for i in range(len(parts)):
            ret.append(delayed(getitem)(result_futures[rank], i))
    del result_futures
    return dask.dataframe.from_delayed(ret, verify_meta=False).persist()


def rearrange_by_column_tasks_wrapper(
    df, column, max_branch=32, npartitions=None, ignore_index=False
):
    """Function wrapper that dispatch the shuffle to explicit-comms.

    Notice, this is monkey patched into Dask at dask_cuda import
    """

    if dask.config.get("explicit-comms", False):
        try:
            import distributed.worker

            # Make sure we have an activate client.
            distributed.worker.get_client()
        except (ImportError, ValueError):
            pass
        else:
            if isinstance(column, str):
                column = [column]
            return dataframe_shuffle(df, column, npartitions, ignore_index)

    from dask.dataframe.shuffle import rearrange_by_column_task_org

    return rearrange_by_column_task_org(
        df, column, max_branch, npartitions, ignore_index
    )
