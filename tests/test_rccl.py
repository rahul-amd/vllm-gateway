#!/usr/bin/env python
"""
Client-side RCCL/NCCL test harness for trl/scripts/vllm_serve.py.

It talks to the FastAPI server over HTTP and uses the same
StatelessProcessGroup + PyNcclCommunicator setup as the server’s
WeightSyncWorkerExtension.

Tests:

  C1: init + close communicator (no collectives)
  C2: small broadcast from client -> workers
  C3: large broadcast from client -> workers (stress)
  C4: lifecycle / double-init sanity checks

If any collective hangs or crashes, that’s a strong signal something
is wrong in RCCL / process-group wiring.

python test_client.py \
  --server-host nid007976 \
  --server-port 8000 \
  --store-host nid007976 \
  --store-port 29500 \
  --device cuda:0 \
  --test all
"""

import argparse
import logging
import sys
from typing import Tuple

import requests
import torch

from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


logger = logging.getLogger("test_client")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_device_uuid(device: str) -> str:
    """
    Return the UUID for the given CUDA/ROCm device.

    On AMD ROCm this is exposed via torch.cuda the same way as on CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available – cannot get device UUID")

    dev = torch.device(device)
    props = torch.cuda.get_device_properties(dev)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        raise RuntimeError(
            "Device properties do not have a 'uuid' field. "
            "This is required by vllm_serve WeightSyncWorkerExtension."
        )
    return str(uuid)


def get_server_world_size(server_url: str) -> int:
    """Call /get_world_size/ on the server."""
    resp = requests.get(f"{server_url}/get_world_size/")
    resp.raise_for_status()
    data = resp.json()
    world_size = int(data["world_size"])
    logger.info("Server reports world_size = %d (TP * DP)", world_size)
    return world_size


def init_weight_sync_communicator(
    server_url: str,
    store_host: str,
    store_port: int,
    device: str,
) -> Tuple[PyNcclCommunicator, int, int]:
    """
    Perform the same init path as the server expects:

    1) Ask server for its world_size (TP * DP).
    2) Compute weight-sync world_size = server_world_size + 1 (client rank).
    3) POST /init_communicator/ so each worker calls init_communicator(...)
       and creates its own PyNcclCommunicator with StatelessProcessGroup.
    4) On the client, create the matching StatelessProcessGroup and
       PyNcclCommunicator with rank = world_size - 1.
    """
    # 1) Get world_size of vLLM engine
    server_world_size = get_server_world_size(server_url)
    weight_sync_world_size = server_world_size + 1
    client_rank = weight_sync_world_size - 1
    logger.info(
        "Weight-sync world_size = server_world_size (%d) + 1 (client) = %d; "
        "client_rank = %d",
        server_world_size,
        weight_sync_world_size,
        client_rank,
    )

    # 2) Get the client GPU UUID (server will assert it's different from its own)
    client_device_uuid = get_device_uuid(device)
    logger.info("Client device %s has UUID %s", device, client_device_uuid)

    # 3) Ask server to initialize its side of the communicator
    init_payload = {
        "host": store_host,
        "port": int(store_port),
        "world_size": int(weight_sync_world_size),
        "client_device_uuid": client_device_uuid,
    }
    logger.info(
        "Calling %s/init_communicator/ with host=%s, port=%d, world_size=%d",
        server_url,
        store_host,
        store_port,
        weight_sync_world_size,
    )
    resp = requests.post(f"{server_url}/init_communicator/", json=init_payload)
    resp.raise_for_status()
    logger.info("Server /init_communicator/ response: %s", resp.json())

    # 4) Create the client-side StatelessProcessGroup + PyNcclCommunicator
    logger.info(
        "Creating StatelessProcessGroup on client (host=%s, port=%d, rank=%d, world_size=%d)",
        store_host,
        store_port,
        client_rank,
        weight_sync_world_size,
    )
    pg = StatelessProcessGroup.create(
        host=store_host,
        port=store_port,
        rank=client_rank,
        world_size=weight_sync_world_size,
    )

    logger.info("Creating PyNcclCommunicator on device %s", device)
    comm = PyNcclCommunicator(pg, device=device)

    logger.info("PyNcclCommunicator successfully initialized.")
    return comm, weight_sync_world_size, client_rank


def close_weight_sync(server_url: str) -> None:
    """Ask server to close its weight-sync communicator."""
    logger.info("Calling %s/close_communicator/", server_url)
    resp = requests.post(f"{server_url}/close_communicator/")
    resp.raise_for_status()
    logger.info("Server /close_communicator/ response: %s", resp.json())


def trigger_server_broadcast(
    server_url: str,
    name: str,
    dtype: str,
    shape,
) -> None:
    """
    Trigger one server-side call to WeightSyncWorkerExtension.update_named_param()
    via the /update_named_param/ endpoint.

    This will cause each worker to:
      1) Allocate an empty tensor of the given shape and dtype.
      2) Perform communicator.broadcast(weight, src=client_rank).
      3) Load the received weights into the model.
    """
    payload = {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
    }
    logger.info(
        "Calling %s/update_named_param/ with name=%s, dtype=%s, shape=%s",
        server_url,
        name,
        dtype,
        list(shape),
    )
    resp = requests.post(f"{server_url}/update_named_param/", json=payload)
    resp.raise_for_status()
    logger.info("Server /update_named_param/ response: %s", resp.json())


# ---------------------------------------------------------------------------
# Tests C1–C4
# ---------------------------------------------------------------------------

def test_C1(server_url: str, store_host: str, store_port: int, device: str) -> None:
    """
    C1: Basic communicator init + close.

    This exercises:
      - TCPStore and StatelessProcessGroup setup across all nodes.
      - PyNcclCommunicator initialization (which internally calls ncclCommInitRank/RCCL).
    """
    logger.info("=== C1: init + close communicator ===")
    comm, world_size, client_rank = init_weight_sync_communicator(
        server_url, store_host, store_port, device
    )
    logger.info(
        "C1: communicator initialized (world_size=%d, client_rank=%d).",
        world_size,
        client_rank,
    )

    # We don’t run any collective here; successful creation already
    # exercises a fair bit of RCCL/NCCL machinery.
    del comm  # drop reference on client side
    close_weight_sync(server_url)
    logger.info("C1: PASSED\n")


def test_C2(
    server_url: str,
    store_host: str,
    store_port: int,
    device: str,
) -> None:
    """
    C2: Small broadcast test.

    Flow:
      1) Init communicator.
      2) Trigger /update_named_param/ with a small shape.
      3) Immediately call communicator.broadcast() on the client with the
         same shape and dtype.

    If RCCL is wired correctly, all ranks (server workers + client) will
    rendezvous in the broadcast and return without error or hang.
    """
    logger.info("=== C2: small broadcast test ===")
    comm, world_size, client_rank = init_weight_sync_communicator(
        server_url, store_host, store_port, device
    )
    logger.info("C2: communicator ready (world_size=%d, client_rank=%d)", world_size, client_rank)

    shape = (4, 4)
    dtype = torch.float32
    tensor = torch.arange(
        start=0,
        end=shape[0] * shape[1],
        dtype=dtype,
        device=device,
    ).reshape(shape)
    logger.info("C2: client broadcast tensor:\n%s", tensor)

    # Ask server to allocate its receive buffers and enter broadcast.
    trigger_server_broadcast(
        server_url,
        name="__rccl_test_small__",
        dtype="torch.float32",
        shape=shape,
    )

    # Now participate in the same broadcast as src=client_rank
    logger.info("C2: entering communicator.broadcast on client (src=%d)", client_rank)
    comm.broadcast(tensor, src=client_rank)
    logger.info("C2: broadcast completed successfully.")

    # Root keeps its own data; no correctness check on server possible from here,
    # but we at least know the collective did not hang or crash.
    logger.info("C2: client tensor after broadcast (should be unchanged on root):\n%s", tensor)

    del comm
    close_weight_sync(server_url)
    logger.info("C2: PASSED\n")


def test_C3(
    server_url: str,
    store_host: str,
    store_port: int,
    device: str,
    large_numel: int,
    num_iters: int,
) -> None:
    """
    C3: Large broadcast stress test.

    Similar to C2 but with a large tensor and multiple iterations to stress:
      - RCCL bandwidth & segmentation
      - communication robustness under load
    """
    logger.info("=== C3: large broadcast stress test ===")
    logger.info(
        "C3: using large_numel=%d (%.2f MiB for float32), num_iters=%d",
        large_numel,
        large_numel * 4 / (1024 ** 2),
        num_iters,
    )

    comm, world_size, client_rank = init_weight_sync_communicator(
        server_url, store_host, store_port, device
    )
    logger.info("C3: communicator ready (world_size=%d, client_rank=%d)", world_size, client_rank)

    shape = (large_numel,)
    dtype = torch.float32

    for it in range(num_iters):
        logger.info("C3: iteration %d / %d", it + 1, num_iters)

        # 1) Prepare client tensor with a simple pattern
        tensor = torch.linspace(
            0.0,
            1.0,
            steps=large_numel,
            dtype=dtype,
            device=device,
        )
        logger.info("C3: prepared client tensor (first 5 elements: %s)", tensor[:5])

        # 2) Trigger server-side broadcast
        trigger_server_broadcast(
            server_url,
            name=f"__rccl_test_large_it{it}__",
            dtype="torch.float32",
            shape=shape,
        )

        # 3) Join the broadcast as root
        logger.info("C3: entering communicator.broadcast on client (src=%d)", client_rank)
        comm.broadcast(tensor, src=client_rank)
        logger.info(
            "C3: broadcast completed (iteration %d). First 5 elements still: %s",
            it + 1,
            tensor[:5],
        )

    del comm
    close_weight_sync(server_url)
    logger.info("C3: PASSED\n")


def test_C4(
    server_url: str,
    store_host: str,
    store_port: int,
    device: str,
) -> None:
    """
    C4: Lifecycle / double-init sanity.

    What this checks:

      - init_communicator() on a clean instance works.
      - Calling /init_communicator/ again while the workers already have a
        communicator should trigger the RuntimeError in WeightSyncWorkerExtension
        (you'll see it in server logs). From the HTTP side, the endpoint still
        returns 200 because it uses 'fire_and_forget', but we can at least
        exercise the path.
      - close_communicator() can be called and the server does not hang.
      - A second close_communicator() call is tolerated.

    NOTE: correctness of the error handling is best inspected in the server logs.
    """
    logger.info("=== C4: lifecycle / double-init test ===")

    # First init – normal path
    comm, world_size, client_rank = init_weight_sync_communicator(
        server_url, store_host, store_port, device
    )
    logger.info("C4: first communicator init OK (world_size=%d, client_rank=%d)", world_size, client_rank)

    # Second init – server side should raise RuntimeError inside workers
    # (see WeightSyncWorkerExtension.init_communicator).
    logger.info("C4: trying second /init_communicator/ (should fail in server worker extension)")

    # We don't re-create a new client communicator, we just call the endpoint
    # to exercise the server code path.
    try:
        # We re-use the same world_size and device UUID for clarity, but the
        # server ignores the provided world_size and recomputes it internally.
        uuid = get_device_uuid(device)
        payload = {
            "host": store_host,
            "port": int(store_port),
            "world_size": int(world_size),
            "client_device_uuid": uuid,
        }
        resp = requests.post(f"{server_url}/init_communicator/", json=payload)
        resp.raise_for_status()
        logger.info("C4: second /init_communicator/ HTTP response: %s", resp.json())
        logger.info(
            "C4: check server logs – you should see 'Weight update group already "
            "initialized. Call close_communicator first.'"
        )
    except Exception as exc:
        logger.warning("C4: second /init_communicator/ raised on HTTP side: %r", exc)

    # Close once
    del comm
    close_weight_sync(server_url)

    # Close twice – should be a no-op from server perspective.
    logger.info("C4: calling /close_communicator/ a second time (should be tolerated)")
    try:
        close_weight_sync(server_url)
    except Exception as exc:
        logger.warning("C4: second /close_communicator/ raised: %r", exc)

    logger.info("C4: PASSED (from client perspective – inspect server logs for details)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCCL/NCCL weight-sync test client for trl/scripts/vllm_serve.py"
    )
    parser.add_argument(
        "--server-host",
        type=str,
        required=True,
        help="Host where vllm_serve FastAPI server is listening.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port of the vllm_serve FastAPI server (default: 8000).",
    )
    parser.add_argument(
        "--store-host",
        type=str,
        help=(
            "Host to use for the TCPStore / StatelessProcessGroup. "
            "If not set, defaults to --server-host."
        ),
    )
    parser.add_argument(
        "--store-port",
        type=int,
        default=29500,
        help=(
            "Port for the TCPStore / StatelessProcessGroup (default: 29500). "
            "Must be reachable from server rank 0 and this client."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use on the client (default: cuda:0). On ROCm this is also cuda:IDX.",
    )
    parser.add_argument(
        "--large-numel",
        type=int,
        default=16 * 1024 * 1024,  # 16M elements ~ 64 MiB float32
        help="Number of elements for large broadcast in C3 (default: 16M).",
    )
    parser.add_argument(
        "--num-large-iters",
        type=int,
        default=3,
        help="Number of iterations for C3 large broadcast (default: 3).",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["C1", "C2", "C3", "C4", "all"],
        default="all",
        help="Which test to run (default: all).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    server_url = f"http://{args.server_host}:{args.server_port}"
    store_host = args.store_host or args.server_host

    logger.info("Server URL:  %s", server_url)
    logger.info("Store host:  %s", store_host)
    logger.info("Store port:  %d", args.store_port)
    logger.info("Client dev:  %s", args.device)

    try:
        if args.test in ("C1", "all"):
            test_C1(server_url, store_host, args.store_port, args.device)
        if args.test in ("C2", "all"):
            test_C2(server_url, store_host, args.store_port, args.device)
        if args.test in ("C3", "all"):
            test_C3(
                server_url,
                store_host,
                args.store_port,
                args.device,
                large_numel=args.large_numel,
                num_iters=args.num_large_iters,
            )
        if args.test in ("C4", "all"):
            test_C4(server_url, store_host, args.store_port, args.device)

        logger.info("All requested tests finished.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error while running tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
