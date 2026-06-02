#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
GPU Availability Checker for SLURM Cluster

Shows GPU availability across all partitions with detailed breakdown
of idle, mixed (partially used), and fully allocated nodes.
"""

import re
import subprocess
from collections import defaultdict


def run_command(cmd):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def parse_node_range(node_str):
    """Expand node range like 'cxis-[0-3,5]' to list of nodes."""
    nodes = []
    # Handle simple case: single node
    if "[" not in node_str:
        return [node_str]

    # Parse ranges like cxis-[0-3,5,7-9]
    prefix_match = re.match(r"([a-zA-Z0-9-]+)\[(.+)\]", node_str)
    if prefix_match:
        prefix = prefix_match.group(1)
        ranges = prefix_match.group(2)
        for part in ranges.split(","):
            if "-" in part:
                start, end = part.split("-")
                for i in range(int(start), int(end) + 1):
                    nodes.append(f"{prefix}{i}")
            else:
                nodes.append(f"{prefix}{part}")
    return nodes


def get_gpu_info():
    """Get detailed GPU information per node."""
    # Get node-level GPU info
    cmd = "sinfo -N -o '%N %P %G %t' --noheader"
    output = run_command(cmd)

    node_info = {}
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            node = parts[0]
            partition = parts[1].rstrip("*")
            gres = parts[2]
            state = parts[3]

            # Parse GPU count from GRES - handle various formats:
            # gpu:H100:8, gpu:tesla:2, gpu:8, gres/gpu:4, etc.
            gpu_type = "GPU"
            gpu_total = 0

            # Try format: gpu:TYPE:COUNT (e.g., gpu:H100:8)
            gpu_match = re.search(r"gpu:([a-zA-Z][a-zA-Z0-9_]*):(\d+)", gres)
            if gpu_match:
                gpu_type = gpu_match.group(1)
                gpu_total = int(gpu_match.group(2))
            else:
                # Try format: gpu:COUNT (e.g., gpu:8)
                gpu_match = re.search(r"gpu:(\d+)", gres)
                if gpu_match:
                    gpu_total = int(gpu_match.group(1))

            if node not in node_info:
                node_info[node] = {
                    "partitions": set(),
                    "gpu_type": gpu_type,
                    "gpu_total": gpu_total,
                    "state": state,
                    "gpu_used": 0,
                }
            node_info[node]["partitions"].add(partition)

    # Get GPU usage per node
    cmd = "sinfo -N -o '%N %G %T' --noheader"
    output = run_command(cmd)

    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            node = parts[0]
            if node in node_info:
                # Update state
                if len(parts) >= 3:
                    node_info[node]["state"] = parts[2]

    # Get actual GPU allocation from squeue
    cmd = "squeue -o '%N %b' --noheader 2>/dev/null"
    output = run_command(cmd)

    gpu_usage = defaultdict(int)
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            nodes_str = parts[0]
            gres = parts[1]

            # Parse GPU count requested
            gpu_match = re.search(r"gpu[:\w]*:?(\d+)", gres)
            gpus_requested = int(gpu_match.group(1)) if gpu_match else 1

            # Expand node list
            for node in parse_node_range(nodes_str):
                gpu_usage[node] += gpus_requested

    # Update node_info with usage
    for node, used in gpu_usage.items():
        if node in node_info:
            node_info[node]["gpu_used"] = used

    return node_info


def get_partition_nodes():
    """Get which nodes belong to which partitions."""
    cmd = "sinfo -o '%P %N' --noheader"
    output = run_command(cmd)

    partition_nodes = defaultdict(list)
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            partition = parts[0].rstrip("*")
            nodes_str = parts[1]
            partition_nodes[partition] = parse_node_range(nodes_str)

    return partition_nodes


def print_summary(node_info, partition_nodes):
    """Print a formatted summary of GPU availability."""

    print("=" * 80)
    print("GPU AVAILABILITY REPORT")
    print("=" * 80)

    # Get current user's jobs
    user = run_command("whoami")
    user_jobs = run_command(f"squeue -u {user} --noheader | wc -l")
    print(f"\nYour running jobs: {user_jobs}")

    # Summary by partition
    print("\n" + "=" * 80)
    print("SUMMARY BY PARTITION")
    print("=" * 80)

    for partition in sorted(partition_nodes.keys()):
        nodes = partition_nodes[partition]

        total_gpus = 0
        available_gpus = 0
        idle_nodes = []
        mixed_nodes = []
        alloc_nodes = []
        down_nodes = []

        for node in nodes:
            if node not in node_info:
                continue

            info = node_info[node]
            gpu_total = info["gpu_total"]
            gpu_used = info["gpu_used"]
            gpu_avail = gpu_total - gpu_used
            state = info["state"].lower()

            total_gpus += gpu_total

            if "down" in state or "drain" in state or "fail" in state:
                down_nodes.append((node, gpu_total, 0))
            elif "idle" in state:
                available_gpus += gpu_total
                idle_nodes.append((node, gpu_total, gpu_total))
            elif "mix" in state:
                available_gpus += gpu_avail
                mixed_nodes.append((node, gpu_total, gpu_avail))
            elif "alloc" in state:
                alloc_nodes.append((node, gpu_total, 0))

        if total_gpus == 0:
            continue

        print(f"\n{'─' * 80}")
        print(f"PARTITION: {partition}")
        print(f"{'─' * 80}")
        print(f"  Total GPUs: {total_gpus} | Available: {available_gpus} ({100 * available_gpus / total_gpus:.1f}%)")
        print(
            f"  Nodes: {len(nodes)} total | {len(idle_nodes)} idle | {len(mixed_nodes)} mixed | {len(alloc_nodes)} allocated | {len(down_nodes)} down/drain"
        )

        if idle_nodes:
            print(f"\n  IDLE NODES ({len(idle_nodes)} nodes, {sum(n[2] for n in idle_nodes)} GPUs available):")
            # Group consecutive nodes
            idle_node_names = [n[0] for n in idle_nodes]
            print(
                f"    {', '.join(idle_node_names[:10])}"
                + (f" ... (+{len(idle_node_names) - 10} more)" if len(idle_node_names) > 10 else "")
            )

        if mixed_nodes:
            # Only show nodes with available GPUs
            mixed_with_avail = [(n, t, a) for n, t, a in mixed_nodes if a > 0]
            mixed_fully_used = len(mixed_nodes) - len(mixed_with_avail)

            print(f"\n  MIXED NODES ({len(mixed_nodes)} nodes, {sum(n[2] for n in mixed_nodes)} GPUs available):")
            if mixed_with_avail:
                for node, total, avail in sorted(mixed_with_avail, key=lambda x: -x[2])[:15]:
                    gpu_type = node_info[node]["gpu_type"]
                    print(f"    {node}: {avail}/{total} {gpu_type} available")
                if len(mixed_with_avail) > 15:
                    print(f"    ... and {len(mixed_with_avail) - 15} more nodes with available GPUs")
            if mixed_fully_used > 0:
                print(f"    ({mixed_fully_used} other mixed nodes have 0 GPUs available)")

        if down_nodes:
            print(f"\n  DOWN/DRAIN NODES: {len(down_nodes)}")

    # Quick summary table
    print("\n" + "=" * 80)
    print("QUICK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Partition':<15} {'Total GPUs':>12} {'Available':>12} {'% Avail':>10}")
    print("-" * 50)

    for partition in sorted(partition_nodes.keys()):
        nodes = partition_nodes[partition]
        total_gpus = 0
        available_gpus = 0

        for node in nodes:
            if node not in node_info:
                continue
            info = node_info[node]
            state = info["state"].lower()

            if "down" not in state and "drain" not in state and "fail" not in state:
                total_gpus += info["gpu_total"]
                if "idle" in state:
                    available_gpus += info["gpu_total"]
                elif "mix" in state:
                    available_gpus += info["gpu_total"] - info["gpu_used"]

        if total_gpus > 0:
            pct = 100 * available_gpus / total_gpus
            print(f"{partition:<15} {total_gpus:>12} {available_gpus:>12} {pct:>9.1f}%")

    print("=" * 80)


def main():
    print("Fetching cluster GPU information...")
    node_info = get_gpu_info()
    partition_nodes = get_partition_nodes()
    print_summary(node_info, partition_nodes)


if __name__ == "__main__":
    main()
