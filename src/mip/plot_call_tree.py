from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from mip.analyze_trace import batch_size_from_name, load_trace, trace_paths
from mip.plot_results import shorten_event_name


@dataclass
class TreeNode:
    id: str
    label: str
    full_name: str
    category: str
    parent: str
    exclusive_us: float = 0.0
    inclusive_us: float = 0.0
    count: int = 0
    children: dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create profiler-style call tree/icicle charts from traces."
    )
    parser.add_argument(
        "trace_input",
        type=Path,
        help="Trace JSON file, .json.gz file, or directory containing traces.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/call_trees"),
        help="Directory for generated HTML call tree charts.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "user_annotation",
            "gpu_user_annotation",
            "cpu_op",
            "cuda_runtime",
            "cuda_driver",
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ],
        help="Trace categories to include in the nested call tree.",
    )
    parser.add_argument(
        "--min-duration-us",
        type=float,
        default=20.0,
        help="Drop events shorter than this duration before tree construction.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Maximum nesting depth to keep in the chart.",
    )
    parser.add_argument(
        "--orientation",
        choices=["h", "v"],
        default="h",
        help="Icicle orientation. Use v to rotate the call tree by 90 degrees.",
    )
    parser.add_argument(
        "--title",
        default="Profiler Call Tree",
        help="Chart title.",
    )
    parser.add_argument(
        "--root-label",
        default=None,
        help="Override the root label shown in the chart.",
    )
    parser.add_argument(
        "--root-duration-ms",
        type=float,
        default=None,
        help="Force the root duration so related charts share a visual time scale.",
    )
    parser.add_argument(
        "--hide-thread-nodes",
        action="store_true",
        help="Attach top-level events directly to the trace root instead of adding pid/tid bars.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Use this filename stem instead of deriving one from the trace filename.",
    )
    return parser.parse_args()


def event_duration_us(event: dict[str, Any]) -> float:
    return float(event.get("dur") or 0.0)


def event_start_us(event: dict[str, Any]) -> float:
    return float(event.get("ts") or 0.0)


def event_end_us(event: dict[str, Any]) -> float:
    return event_start_us(event) + event_duration_us(event)


def thread_key(event: dict[str, Any]) -> tuple[object, object]:
    return event.get("pid"), event.get("tid")


def thread_label(pid: object, tid: object) -> str:
    return f"pid={pid} tid={tid}"


def sanitized_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name.replace(".json.gz", "").replace(".json", ""))


def add_child_node(
    nodes: dict[str, TreeNode],
    parent_id: str,
    *,
    name: str,
    category: str,
) -> str:
    parent = nodes[parent_id]
    key = f"{category}:{name}"
    existing = parent.children.get(key)
    if existing is not None:
        return existing

    node_id = f"{parent_id}/{len(parent.children)}"
    nodes[node_id] = TreeNode(
        id=node_id,
        label=shorten_event_name(name, max_length=56),
        full_name=name,
        category=category,
        parent=parent_id,
    )
    parent.children[key] = node_id
    return node_id


def build_call_tree(
    trace_path: Path,
    *,
    categories: set[str],
    min_duration_us: float,
    max_depth: int,
    root_label: str | None = None,
    root_duration_us: float | None = None,
    hide_thread_nodes: bool = False,
) -> list[TreeNode]:
    trace = load_trace(trace_path)
    grouped_events: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(list)

    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        if event.get("cat") not in categories:
            continue
        if event_duration_us(event) < min_duration_us:
            continue
        grouped_events[thread_key(event)].append(event)

    batch_size = batch_size_from_name(trace_path)
    default_root_label = f"batch_size={batch_size}" if batch_size is not None else trace_path.name
    nodes: dict[str, TreeNode] = {
        "root": TreeNode(
            id="root",
            label=root_label or default_root_label,
            full_name=str(trace_path),
            category="trace",
            parent="",
        )
    }

    for (pid, tid), events in grouped_events.items():
        events.sort(key=lambda event: (event_start_us(event), -event_duration_us(event)))
        thread_id = "root"
        if not hide_thread_nodes:
            thread_id = add_child_node(
                nodes,
                "root",
                name=thread_label(pid, tid),
                category="thread",
            )
        stack: list[tuple[str, float, float]] = [(thread_id, float("inf"), 0.0)]

        for event in events:
            start_us = event_start_us(event)
            end_us = event_end_us(event)
            duration_us = event_duration_us(event)
            while len(stack) > 1 and start_us >= stack[-1][1]:
                stack.pop()

            parent_id = stack[-1][0]
            if len(stack) > max_depth:
                parent_id = stack[max_depth][0]
            node_id = add_child_node(
                nodes,
                parent_id,
                name=str(event.get("name") or "unknown"),
                category=str(event.get("cat") or "uncategorized"),
            )
            nodes[node_id].inclusive_us += duration_us
            nodes[node_id].exclusive_us += duration_us
            nodes[node_id].count += 1

            if stack:
                nodes[stack[-1][0]].exclusive_us = max(
                    nodes[stack[-1][0]].exclusive_us - duration_us,
                    0.0,
                )
            stack.append((node_id, end_us, duration_us))

    def fill_container_totals(node_id: str) -> tuple[float, int]:
        node = nodes[node_id]
        child_totals = [fill_container_totals(child_id) for child_id in node.children.values()]
        child_duration_us = sum(duration_us for duration_us, _ in child_totals)
        child_count = sum(count for _, count in child_totals)
        if node.category in {"trace", "thread"}:
            node.inclusive_us = child_duration_us
            node.exclusive_us = 0.0
            node.count = max(1, child_count)
        return node.inclusive_us, node.count

    fill_container_totals("root")
    if root_duration_us is not None:
        nodes["root"].inclusive_us = max(nodes["root"].inclusive_us, root_duration_us)
    return list(nodes.values())


def write_call_tree(
    nodes: list[TreeNode],
    output_path: Path,
    *,
    title: str,
    orientation: str,
) -> None:
    filtered_nodes = [
        node
        for node in nodes
        if node.id == "root" or node.category == "thread" or node.inclusive_us > 0
    ]
    ids = [node.id for node in filtered_nodes]
    labels = [node.label for node in filtered_nodes]
    parents = [node.parent for node in filtered_nodes]
    values = [max(node.inclusive_us / 1000, 0.001) for node in filtered_nodes]
    custom_data = [
        [
            node.full_name,
            node.category,
            node.count,
            node.inclusive_us / 1000,
            node.exclusive_us / 1000,
        ]
        for node in filtered_nodes
    ]
    palette = {
        "trace": "#efe7dc",
        "thread": "#d8ddd2",
        "user_annotation": "#8a5b49",
        "gpu_user_annotation": "#8a5b49",
        "cpu_op": "#486f73",
        "cuda_runtime": "#6f5f8f",
        "cuda_driver": "#786d58",
        "kernel": "#b66a4d",
        "gpu_memcpy": "#6b856d",
        "gpu_memset": "#6b856d",
    }
    colors = [palette.get(node.category, "#9aa69f") for node in filtered_nodes]

    fig = go.Figure(
        go.Icicle(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            customdata=custom_data,
            marker={"colors": colors},
            tiling={"orientation": orientation},
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "Count: %{customdata[2]}<br>"
                "Inclusive time: %{customdata[3]:.4f} ms<br>"
                "Exclusive self time: %{customdata[4]:.4f} ms"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        margin={"l": 16, "r": 16, "t": 48, "b": 12},
        height=680,
        font={"family": "Inter, Arial, sans-serif", "color": "#172026"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    paths = trace_paths(args.trace_input)
    if not paths:
        raise SystemExit(f"No trace files found at {args.trace_input}")

    for path in paths:
        nodes = build_call_tree(
            path,
            categories=set(args.categories),
            min_duration_us=args.min_duration_us,
            max_depth=args.max_depth,
            root_label=args.root_label,
            root_duration_us=(
                args.root_duration_ms * 1000 if args.root_duration_ms is not None else None
            ),
            hide_thread_nodes=args.hide_thread_nodes,
        )
        write_call_tree(
            nodes,
            args.output_dir / f"{args.output_stem or sanitized_stem(path)}_call_tree.html",
            title=args.title,
            orientation=args.orientation,
        )


if __name__ == "__main__":
    main()
