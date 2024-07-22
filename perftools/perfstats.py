import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Union

from .logger import logger
from .perfdata import AggregatedStackFrame, Perfdata, Thread

StrPath = Union[str, os.PathLike]


class StatsNode:
    """Stats Node, a container to store stats values for time frames."""

    @staticmethod
    def create(thread: Thread) -> "StatsNode":
        """Create stats node from thread."""

        root = StatsNode(thread.thread_name)

        agg_frames = thread.aggregate_frames()[1:-1]  # drop first and last frame

        __count = 0
        __last_time = time.time()
        for frame in agg_frames:
            root.merge_frame(frame)

            __count += 1
            __now_time = time.time()
            if __now_time - __last_time > 10:
                __last_time = __now_time
                logger.debug("Thread %s: %d frames processed", thread.unique_name, __count)

        logger.info("Thread %s: %d frames processed totally.", thread.unique_name, __count)
        return root

    def __init__(self, name: str, father_node: Optional["StatsNode"] = None) -> None:
        self.name = name
        self.father_node = father_node
        self.child_nodes: Dict[str, StatsNode] = {}

        self.duration = 0
        self.call_count = 0
        self._max_duration = 0
        self._max_call_count = 0

    @property
    def frame_count(self) -> int:
        """Returns call count of root node."""
        node = self
        while node.father_node is not None:
            node = node.father_node
        return node.call_count

    @property
    def max_duration(self) -> float:
        return self._max_duration

    @max_duration.setter
    def max_duration(self, value: float):
        self._max_duration = max(self._max_duration, value)

    @property
    def max_call_count(self) -> int:
        return self._max_call_count

    @max_call_count.setter
    def max_call_count(self, value: int):
        self._max_call_count = max(self._max_call_count, value)

    @property
    def avg_duration(self) -> float:
        return self.duration / self.call_count

    @property
    def avg_call_count(self) -> float:
        return self.call_count / self.frame_count

    @property
    def global_avg_duration(self) -> float:
        return self.duration / self.frame_count

    @property
    def global_avg_call_count(self) -> float:
        return self.call_count / self.frame_count

    def merge_frame(self, agg_frame: AggregatedStackFrame):
        """Merge an AggregatedStackFrame and update self nodes and values.

        Raises:
            ValueError: Different root names.
        """

        if agg_frame.symbol_name != self.name:
            raise ValueError(f"{agg_frame.symbol_name} can't merge to {self.name}")

        # update self data
        self.duration += agg_frame.duration
        self.call_count += agg_frame.call_count
        self.max_duration = agg_frame.duration
        self.max_call_count = agg_frame.call_count

        # merge child node
        for agg_child_name, agg_child in agg_frame.child_frames.items():
            if agg_child_name not in self.child_nodes:
                self.child_nodes[agg_child_name] = StatsNode(agg_child_name, self)
            self.child_nodes[agg_child_name].merge_frame(agg_child)
