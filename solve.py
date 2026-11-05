#!/usr/bin/env python3
"""Run the exact LNS-RL-V1 policy evaluated in the KES 2025 paper."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from lib.ppo_1 import Agent
from lib.search_animation import save_search_animation
from lib.vrp_helper_1 import get_distMat, parse_cvrp_answer, parse_cvrp_question
from lib.vrp_helper_env import build_edge_index_near, build_edge_index_routes, vrp_env_sisr


ROOT = Path(__file__).resolve().parent
PAPER_CHECKPOINT = ROOT / "models" / "lns_rl_kes.pth"
PAPER_CHECKPOINT_SHA256 = "ffcf8f83546a665c3cecd8d0b0a904a9d0ce3854a75dd12636fb3239b3efb664"


def _write_array(data: np.ndarray, path: Path) -> None:
    path.write_text("".join(" ".join(map(str, row)) + "\n" for row in data))


def _write_distances(matrix: np.ndarray, path: Path) -> None:
    line = ";".join(",".join(map(str, np.round(row, 4))) for row in matrix)
    path.write_text(line + "\n")


def _serialize_state(
    capacity: float, ruins: list[int], routes: list[list[int]]
) -> str:
    route_text = ";".join(",".join(map(str, route)) for route in routes)
    return f"{capacity}:{','.join(map(str, ruins))}:{route_text}"


def _parse_java_output(output: str) -> tuple[float, list[list[int]]]:
    pattern = re.compile(r"^\s*([-+0-9.eE]+):(.*)\s*$")
    for raw_line in reversed(output.splitlines()):
        match = pattern.match(raw_line)
        if not match:
            continue
        routes = [
            [int(node) for node in route.split(",") if node]
            for route in match.group(2).split(";")
            if route
        ]
        return float(match.group(1)), routes
    raise RuntimeError(f"Java solver returned no parseable solution: {output!r}")


def _compile_java() -> None:
    java_dir = ROOT / "JAVA"
    sources = [
        java_dir / "ArrayIndexComparator.java",
        java_dir / "RandomSolver.java",
        java_dir / "SISRSolver.java",
        java_dir / "Recreate.java",
        java_dir / "RepairWorker.java",
    ]
    class_files = [java_dir / f"{source.stem}.class" for source in sources]
    if all(path.exists() for path in class_files):
        newest_source = max(path.stat().st_mtime_ns for path in sources)
        oldest_class = min(path.stat().st_mtime_ns for path in class_files)
        if oldest_class >= newest_source:
            return
    subprocess.run(
        ["javac", "-classpath", str(java_dir), *(str(path) for path in sources)],
        check=True,
    )


def _run_java(class_name: str, *arguments: object) -> str:
    completed = subprocess.run(
        ["java", "-classpath", str(ROOT / "JAVA"), class_name, *(str(arg) for arg in arguments)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


class _RepairWorker:
    """Long-lived Java process for the unchanged regret-repair operator."""

    def __init__(self, data_path: Path, distance_path: Path) -> None:
        self.process = subprocess.Popen(
            [
                "java",
                "-classpath",
                str(ROOT / "JAVA"),
                "RepairWorker",
                str(data_path),
                str(distance_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def repair(
        self, capacity: float, ruins: list[int], routes: list[list[int]]
    ) -> tuple[float, list[list[int]]]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Java repair worker has no communication pipes")
        request = _serialize_state(capacity, ruins, routes)
        try:
            self.process.stdin.write(request + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(self._failure_message()) from error

        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(self._failure_message())
        if response.startswith("ERROR:"):
            raise RuntimeError(f"Java repair worker failed: {response.strip()}")
        return _parse_java_output(response)

    def _failure_message(self) -> str:
        details = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            details = self.process.stderr.read().strip()
        message = "Java repair worker stopped unexpectedly"
        return f"{message}: {details}" if details else message

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)

    def __enter__(self) -> _RepairWorker:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solve(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = Path(args.model).resolve()
    if checkpoint == PAPER_CHECKPOINT.resolve():
        actual_hash = _sha256(checkpoint)
        if actual_hash != PAPER_CHECKPOINT_SHA256:
            raise RuntimeError(
                "The bundled paper checkpoint has changed: "
                f"expected {PAPER_CHECKPOINT_SHA256}, got {actual_hash}."
            )

    capacity, instance_data = parse_cvrp_question(args.instance)
    n_nodes = len(instance_data)
    n_customers = n_nodes - 1
    if n_customers >= 200:
        raise ValueError(
            "The KES evaluation code only evaluated instances with fewer than "
            f"200 customers; got {n_customers}."
        )

    n_ruins = int(math.ceil(max(math.sqrt(n_customers), 1.0) * 1.2 / args.anchors))
    n_ruins_total = n_ruins * args.anchors
    alpha_t = (args.final_temperature / args.initial_temperature) ** (1.0 / args.iterations)

    _compile_java()
    with tempfile.TemporaryDirectory(prefix="lns_rl_kes_") as temp_name:
        temp_dir = Path(temp_name)
        data_path = temp_dir / "data_0.txt"
        distance_path = temp_dir / "distances.txt"
        _write_array(instance_data, data_path)

        initial_output = _run_java(
            "RandomSolver", data_path, int(capacity), 0, n_ruins_total
        )
        initial_distance, initial_routes = _parse_java_output(initial_output)
        animation_every = args.animation_every or max(1, args.iterations // 60)
        snapshots = [
            (0, float(initial_distance), copy.deepcopy(initial_routes), 0.0)
        ] if args.animation else []
        distance_matrix = get_distMat(instance_data)
        _write_distances(distance_matrix, distance_path)

        environment = vrp_env_sisr(
            capacity,
            instance_data,
            distance_matrix,
            initial_routes,
            initial_distance,
            args.initial_temperature,
            alpha_t,
        )

        agent = Agent(
            2e-5,
            (0.9, 0.999),
            0.99,
            1,
            0.2,
            1,
            11,
            128,
            256,
            args.anchors,
            device=device,
        )
        agent.load(str(checkpoint))
        agent.policy_old.eval()

        edge_near = torch.from_numpy(
            build_edge_index_near([distance_matrix], 10)[0]
        ).long().to(device)
        hidden = torch.zeros((1, 128), dtype=torch.float32, device=device)
        previous_input = torch.zeros(
            (1, n_ruins_total, 11), dtype=torch.float32, device=device
        )
        previous_ruins: list[int] = []
        accepted_moves = 0
        improving_moves = 0

        with _RepairWorker(data_path, distance_path) as repair_worker:
            search_started = time.perf_counter()
            for step in range(args.iterations):
                embedding = environment.get_embedding_cvrp_metrics()
                if step:
                    previous_input = torch.from_numpy(
                        (embedding[previous_ruins] * float(was_accepted)).astype(np.float32)
                    ).unsqueeze(0).to(device)

                graph = Data(
                    x=torch.from_numpy(embedding).float().to(device),
                    edge_index_r0=torch.from_numpy(
                        build_edge_index_routes(
                            [environment.curr_routes], inverse=False
                        )[0]
                    ).long().to(device),
                    edge_index_r1=torch.from_numpy(
                        build_edge_index_routes(
                            [environment.curr_routes], inverse=True
                        )[0]
                    ).long().to(device),
                    edge_index_n=edge_near,
                    state_=hidden[0],
                    input_=previous_input[0],
                )
                with torch.inference_mode():
                    hidden, action_node, action_sisr, _node_dist, _sisr_dist = (
                        agent.policy_old.act(graph, n_nodes)
                    )

                coefficients = action_sisr[0, action_node[0]].detach().cpu().numpy()
                previous_ruins = []
                for anchor_index in range(args.anchors):
                    anchor = int(action_node[0, anchor_index].item()) + 1
                    selected = environment.get_ruins(
                        anchor,
                        float(coefficients[anchor_index]),
                        n_ruins,
                        ruined=previous_ruins,
                    )
                    previous_ruins.extend(selected)

                new_distance, new_routes = repair_worker.repair(
                    capacity, previous_ruins, environment.curr_routes
                )
                was_accepted, was_improving, _reward = environment.sim_annealing(
                    new_distance, new_routes
                )
                accepted_moves += int(was_accepted)
                improving_moves += int(was_improving)
                completed_iterations = step + 1
                if args.animation and (
                    completed_iterations % animation_every == 0
                    or completed_iterations == args.iterations
                ):
                    snapshots.append(
                        (
                            completed_iterations,
                            float(environment.best_dist),
                            copy.deepcopy(environment.best_routes),
                            time.perf_counter() - search_started,
                        )
                    )
            search_time_seconds = time.perf_counter() - search_started

    if args.animation:
        save_search_animation(
            instance_data,
            snapshots,
            args.animation,
            args.iterations,
        )

    result = {
        "instance": str(Path(args.instance).resolve()),
        "customers": n_customers,
        "capacity": float(capacity),
        "iterations": args.iterations,
        "anchors": args.anchors,
        "initial_temperature": args.initial_temperature,
        "final_temperature": args.final_temperature,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "model": str(checkpoint),
        "model_sha256": _sha256(checkpoint),
        "initial_distance": float(initial_distance),
        "best_distance": float(environment.best_dist),
        "accepted_moves": accepted_moves,
        "improving_moves": improving_moves,
        "search_time_seconds": search_time_seconds,
        "routes": [[int(node) for node in route] for route in environment.best_routes],
    }
    if args.animation:
        result["animation"] = str(Path(args.animation).resolve())
    if args.solution:
        reference_cost, _routes = parse_cvrp_answer(args.solution)
        result["reference_cost"] = float(reference_cost)
        result["gap_percent"] = (
            100.0 * (float(environment.best_dist) - float(reference_cost)) / float(reference_cost)
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, help="CVRPLIB .vrp instance")
    parser.add_argument("--solution", help="optional matching CVRPLIB .sol file")
    parser.add_argument("--model", default=str(PAPER_CHECKPOINT))
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="LNS iterations; 5000 is the paper setting",
    )
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--initial-temperature", type=float, default=100.0)
    parser.add_argument("--final-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads used by PyTorch; 1 is fastest for the small paper graphs",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="results/generated/solution.json")
    parser.add_argument(
        "--animation",
        help="optional destination for an animated GIF of the best solution",
    )
    parser.add_argument(
        "--animation-every",
        type=int,
        help="capture one GIF frame every N iterations (default: about 60 frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.animation_every is not None and args.animation_every <= 0:
        raise ValueError("--animation-every must be positive")
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    result = solve(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
