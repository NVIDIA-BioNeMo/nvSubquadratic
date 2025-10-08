# TODO: Add license header here

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.distributed.nn.functional import all_gather as functional_all_gather


def init_parallel_state(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
) -> int:
    """Initialize distributed training and megatron parallel state.

    Sets up the distributed training environment using NCCL backend and initializes
    Megatron's parallel state with the specified parallelism configurations. This
    function handles device assignment, process group initialization, and parallel
    state setup.

    Args:
        tensor_model_parallel_size: Number of GPUs for tensor parallelism (default: 1).
        pipeline_model_parallel_size: Number of stages for pipeline parallelism (default: 1).
        context_parallel_size: Number of GPUs for context parallelism (default: 1).

    Returns:
        int: The local rank of the current process.

    Raises:
        AssertionError: If the number of available GPUs doesn't match the required
            world size (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size).

    Note:
        This function sets up environment variables for NCCL configuration and
        initializes the process group if not already initialized. It also verifies
        the context parallel rank and world size after initialization.
    """
    num_gpus = torch.cuda.device_count()
    required_world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    assert num_gpus == required_world_size, (
        f"World size {num_gpus} != TP={tensor_model_parallel_size} x PP={pipeline_model_parallel_size} x CP={context_parallel_size}"
    )

    # Set up environment variables
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Get local rank
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)

    # Set up timeout
    timeout_seconds = int(os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800))
    timeout_timedelta = timedelta(seconds=timeout_seconds)

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout_timedelta)
        logging.info(f"Initialized distributed training with local rank {local_rank}")

    # Initialize parallel state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    # Verify initialization
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_world_size = parallel_state.get_context_parallel_world_size()
    logging.info(f"CP rank: {cp_rank}, CP world size: {cp_world_size}")
    return local_rank


def zigzag_split_across_group_ranks(data, group, seq_dim=0):
    """Distributes tensor data across group ranks using zigzag pattern.

    Divides the input tensor along sequence dimension and distributes chunks
    in an alternating pattern across different ranks.

    Arguments:
        data: original tensor to split across group ranks.
        group: the group to distribute the data across.
        seq_dim: the sequence/context dimension to split.

    Returns:
        Tensor slice for the current rank following zigzag distribution.
    """
    # Get group information
    process_count = len(dist.get_process_group_ranks(group))
    current_rank = dist.get_rank(group)

    # Skip distribution for single process
    if process_count == 1:
        return data

    # Calculate number of chunks for zigzag distribution
    total_chunks = 2 * process_count

    # Divide data into equal chunks
    tensor_chunks = list(torch.chunk(data, total_chunks, dim=seq_dim))

    # Implement zigzag distribution logic:
    # Each rank gets two chunks in specific positions
    # First chunk is at position equal to rank
    first_chunk_idx = current_rank
    # Second chunk is from the end, offset by rank+1
    second_chunk_idx = total_chunks - 1 - current_rank

    # Combine the appropriate chunks for this rank
    rank_data = torch.cat([tensor_chunks[first_chunk_idx], tensor_chunks[second_chunk_idx]], dim=seq_dim)

    return rank_data.contiguous()


def zigzag_gather_from_group_ranks(data, group, seq_dim=0):
    """Reconstructs complete tensor from zigzag-distributed chunks.

    Takes data distributed across ranks in zigzag pattern and reassembles
    the original complete tensor.

    Arguments:
        data: tensor fragment from current rank to be gathered.
        group: the group to gather data from.
        seq_dim: dimension along which to concatenate fragments.

    Returns:
        Reconstructed tensor with fragments from all ranks.
    """
    # Get group information
    process_count = len(dist.get_process_group_ranks(group))

    # Skip gathering for single process
    if process_count == 1:
        return data

    # Gather from all ranks using autograd-enabled all_gather
    gathered_data = functional_all_gather(data, group=group)

    # Initialize a list to store the original sequence chunks with proper tensor type
    seq_chunks = []
    for i in range(2 * process_count):
        seq_chunks.append(None)  # Will be replaced with tensors

    # Process each gathered tensor
    for i, data_i in enumerate(gathered_data):
        chunk_size = data_i.size(seq_dim) // 2

        # Split the data_i back into the original two chunks
        chunk0, chunk1 = torch.split(data_i, chunk_size, dim=seq_dim)

        # Reassign the chunks to their original positions
        seq_chunks[i] = chunk0
        seq_chunks[-(i + 1)] = chunk1

    # Concatenate all chunks to reconstruct the original data
    reconstructed_data = torch.cat(seq_chunks, dim=seq_dim)

    return reconstructed_data


def setup_rank0_logging(log_file: str | None = None) -> None:
    """Set up logging that only prints to console from rank 0, but logs all ranks to files.

    This is useful for distributed training where you want clean console output
    without duplicate messages from each rank, but still want detailed logs
    from all ranks saved to separate files.

    Args:
        log_file: Path to log file. If None, uses /tmp/rank_{rank}.log.
            The actual filename will have the rank appended (e.g., rank_0.log).

    Example:
        >>> from nvsubquadratic.parallel.utils import setup_rank0_logging
        >>> setup_rank0_logging("/tmp/my_test.log")
        >>> logging.info("This only prints to console from rank 0")
    """
    rank = int(os.getenv("RANK", "0"))

    # Determine log file path
    if log_file is None:
        log_file = f"/tmp/rank_{rank}.log"

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Add filter to console handler to only show rank 0 logs
    class Rank0Filter(logging.Filter):
        def filter(self, record):
            return int(os.getenv("RANK", "0")) == 0

    console_handler.addFilter(Rank0Filter())

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler],
        force=True,  # Override any existing configuration
    )
