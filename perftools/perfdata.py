"""perf.data parser module."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import simpleperf_report_lib as reportlib

from .logger import logger

StrPath = Union[str, os.PathLike]


class Sample:
    def __init__(self, sample: reportlib.SampleStruct) -> None:
        self._sample = sample

        self.ip: int = self._sample.ip
        self.pid: int = self._sample.pid
        self.tid: int = self._sample.tid
        self.thread_name: str = self._sample.thread_comm
        self.time: int = self._sample.time
        self.in_kernel: bool = self._sample.in_kernel
        self.cpu: int = self._sample.cpu
        self.period: int = self._sample.period

    def json(self) -> dict:
        return {
            "ip": self.ip,
            "pid": self.pid,
            "tid": self.tid,
            "thread_name": self.thread_name,
            "time": self.time,
            "in_kernel": self.in_kernel,
            "cpu": self.cpu,
            "period": self.period,
        }


class Event:
    def __init__(self, event: reportlib.EventStruct) -> None:
        self._event = event

        self.name: str = self._event.name

    def json(self) -> dict:
        return {
            "name": self.name
        }


class Symbol:
    def __init__(self, symbol: reportlib.SymbolStruct) -> None:
        self._symbol = symbol

        self.dso_name: str = self._symbol.dso_name
        self.vaddr_in_file: int = self._symbol.vaddr_in_file
        self.symbol_name: str = self._symbol.symbol_name
        self.symbol_addr: int = self._symbol.symbol_addr
        self.symbol_len: int = self._symbol.symbol_len

    def json(self) -> dict:
        return {
            "dso_name": self.dso_name,
            "vaddr_in_file": self.vaddr_in_file,
            "symbol_name": self.symbol_name,
            "symbol_addr": self.symbol_addr,
            "symbol_len": self.symbol_len,
        }


class CallChainEntry:
    def __init__(self, call_chain_entry: reportlib.CallChainEntryStructure) -> None:
        self._call_chain_entry = call_chain_entry

        self.ip: int = self._call_chain_entry.ip
        self.symbol = Symbol(self._call_chain_entry.symbol)

    def json(self) -> dict:
        return {
            "ip": self.ip,
            "symbol": self.symbol.json()
        }


class CallChain:
    def __init__(self, call_chain: reportlib.CallChainStructure) -> None:
        self._call_chain = call_chain

        self.num_entries: int = self._call_chain.nr
        self.entries = [CallChainEntry(self._call_chain.entries[i]) for i in range(self._call_chain.nr)]

    def __len__(self) -> int:
        return self.num_entries

    def json(self) -> dict:
        return {
            "num_entries": self.num_entries,
            "entries": [v.json() for v in self.entries]
        }


class SampleInfo:
    """Aggregated sample information.

    Including four mainly used attributes: sample, event, symbol and call_chain
    """

    def __init__(self,
                 sample: reportlib.SampleStruct,
                 event: reportlib.EventStruct,
                 symbol: reportlib.SymbolStruct,
                 call_chain: reportlib.CallChainStructure) -> None:
        self.sample = Sample(sample)
        self.event = Event(event)
        self.symbol = Symbol(symbol)
        self.call_chain = CallChain(call_chain)

    def __str__(self) -> str:
        values = ", ".join([
            f"thread={self.sample.thread_name}-{self.sample.tid}",
            f"symbol={self.symbol.symbol_name}",
            f"time={self.sample.time}",
            f"period=({self.sample.period})"
        ])
        return f"SampleInfo({values})"

    def json(self) -> dict:
        return {
            "sample": self.sample.json(),
            "event": self.event.json(),
            "symbol": self.symbol.json(),
            "call_chain": self.call_chain.json()
        }


class StackFrame:
    """Stack frame used in threads.

    Each stack frame is a Bidirectional Multi-way Tree.
        Use father_frame to get father node, and child_frames to get children nodes.

    The unit of time is nanosecond (ns, 10^-9 second)
    """

    def __init__(self, symbol_name: str, start_time: int, end_time: int, father_frame: Optional["StackFrame"] = None) -> None:
        self.symbol_name = symbol_name
        self.start_time = start_time
        self.end_time = end_time
        self.father_frame = father_frame

        self.child_frames: List[StackFrame] = []

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    def json(self) -> dict:
        if len(self.child_frames) <= 0:
            return None

        k = f"{self.symbol_name}[{self.start_time / 1_000_000:.3f},{self.end_time / 1_000_000:.3f}]"
        v = [c.json() for c in self.child_frames]
        return {k: v}


class AggregatedStackFrame:
    """Aggregated stack frame for threads.

    Each stack frame is a Bidirectional Multi-way Tree.
        Use father_frame to get father node, and child_frames to get children nodes.

    raw_stack_frames stores all aggregated raw stack frames.

    The unit of time is nanosecond (ns, 10^-9 second)
    """

    def __init__(self, stack_frame: StackFrame, father_frame: Optional["AggregatedStackFrame"] = None) -> None:
        self.symbol_name = stack_frame.symbol_name
        self.raw_stack_frames: List[StackFrame] = [stack_frame]
        self.father_frame = father_frame
        self.child_frames: Dict[str, AggregatedStackFrame] = {}

    @property
    def duration(self) -> int:
        return sum(sf.duration for sf in self.raw_stack_frames)

    @property
    def call_count(self) -> int:
        return len(self.raw_stack_frames)

    @property
    def avg_duration(self) -> float:
        return self.duration / self.call_count

    def get_frame_metrics(self, frame_timestamps: Optional[List[Tuple[int, int]]] = None) -> dict:
        """Get game frame related metrics.

        Returns:
            {
                "avg_call_count": total call count / total frames count,
                "max_duration": maximum duration in one frame,
                "max_call_count": maximum call count in one frame,
                "global_avg_duration": total duration / total frames count
            }

            If frame_timestamps not given, return -1 for all values.
        """

        if frame_timestamps is None:
            return {
                "avg_call_count": -1,
                "max_duration": -1,
                "max_call_count": -1,
                "global_avg_duration": -1
            }
        else:
            raise NotImplementedError

    def json(self) -> dict:
        if len(self.child_frames) <= 0:
            return None

        k = f"{self.symbol_name}[{self.duration / 1_000_000:.3f},{self.call_count}]"
        v = [c.json() for c in sorted(self.child_frames.values(), key=lambda x: x.avg_duration, reverse=True)]
        return {k: v}


class Thread:
    def __init__(self, name: str, tid: int) -> None:
        """Thread

        Args:
            name: Thread name, such as UnityMain or GameThread
            tid: Thread id, sampled by simpleperf.
        """

        self.thread_name = name
        self.tid = tid
        self.unique_name = f"{name}-{tid}"

        self.stack_frames: List[StackFrame] = []
        # self.samples = []  # maybe can store samples
        self.samples_count = 0

        self.__tmp_node_count = 0

    @property
    def start_time(self) -> int:
        """Returns start time of first top frame, returns -1 if no frames."""

        if len(self.stack_frames) <= 0:
            return -1
        return self.stack_frames[0].start_time

    @property
    def end_time(self) -> int:
        """Returns end time of last top frame, returns -1 if no frames."""

        if len(self.stack_frames) <= 0:
            return -1
        return self.stack_frames[-1].end_time

    def _create_stack_frame(self, symbols: List[Symbol], start_time: int, end_time: int) -> StackFrame:
        """Create a new stack frame from symbols, the symbols do calls from first to last."""

        top_node = StackFrame(symbols[0].symbol_name, start_time, end_time)

        node = top_node
        for symbol in symbols[1:]:
            sub_node = StackFrame(symbol.symbol_name, start_time, end_time, node)
            node.child_frames.append(sub_node)
            node = sub_node

        return top_node

    def append_sample(self, sample_info: SampleInfo):
        """Append a sample to the thread tail."""

        self.samples_count += 1

        symbols = [sample_info.symbol]
        for s in sample_info.call_chain.entries:
            symbols.append(s.symbol)
        symbols.reverse()

        end_time = sample_info.sample.time
        start_time = (end_time - 1) if self.end_time < 0 else self.end_time

        if end_time < self.end_time:
            logger.warning("%s skipped", sample_info)
            return

        # a newly created thread
        if len(self.stack_frames) <= 0:
            new_stack_frame = self._create_stack_frame(symbols, start_time, end_time)
            self.stack_frames.append(new_stack_frame)
            return

        # try merge stack frames
        current_stack_frame = self.stack_frames[-1]
        top_end_time = current_stack_frame.end_time
        for i, symbol in enumerate(symbols):
            # if has same symbol name, try to merge it
            # for sub frames, current_stack_frame.end_time may less than top_end_time
            # which means that the two have the same father frame, but there is a gap between themselves.
            # currently, we consider them as different calls, even they have the same symbols
            if (
                current_stack_frame.symbol_name == symbol.symbol_name
                and current_stack_frame.end_time == top_end_time
            ):
                # the same symbols and calls, we extend the time
                current_stack_frame.end_time = end_time

                # continue to compare sub stack frames
                if len(current_stack_frame.child_frames) > 0:
                    current_stack_frame = current_stack_frame.child_frames[-1]

                # current stack frames count less than the sample
                # make the rest symbols a sub frame to current stack frame
                elif i + 1 <= len(symbols) - 1:
                    sub_stack_frame = self._create_stack_frame(symbols[i+1:], start_time, end_time)
                    sub_stack_frame.father_frame = current_stack_frame
                    current_stack_frame.child_frames.append(sub_stack_frame)
                    return

            # a new sub call to current father frame
            else:
                sub_stack_frame = self._create_stack_frame(symbols[i:],
                                                           current_stack_frame.end_time,  # maybe we can use top_end_time or start_time?
                                                           end_time)
                father_frame = current_stack_frame.father_frame

                # a new root frame
                if father_frame is None:
                    self.stack_frames.append(sub_stack_frame)

                # a new sub frame of current father frame
                else:
                    sub_stack_frame.father_frame = father_frame
                    father_frame.child_frames.append(sub_stack_frame)

                return

    def _aggregate_all(self, root: Optional[AggregatedStackFrame], child_frames: List[StackFrame]):
        """Generate AggregatedStackFrame recursively. Modify root inplace."""

        agg_child_frames = root.child_frames

        for frame in child_frames:
            if frame.symbol_name not in agg_child_frames:
                agg_child_frames[frame.symbol_name] = AggregatedStackFrame(frame, root)
            else:
                agg_child_frames[frame.symbol_name].raw_stack_frames.append(frame)

        self.__tmp_node_count += 1
        if self.__tmp_node_count % 10000 == 0:
            logger.debug("%d nodes processed.", self.__tmp_node_count)

        for agg_frame in agg_child_frames.values():
            child_frames = []
            for raw_frame in agg_frame.raw_stack_frames:
                child_frames.extend(raw_frame.child_frames)
            self._aggregate_all(agg_frame, child_frames)

    def aggregate(self, use_unique_thread_name: bool = False) -> AggregatedStackFrame:
        """Aggregate all stack frames to a root AggregatedStackFrame, the thread will be seen as a function call.

        Args:
            use_unique_thread_name: If true, set root thread name with thread id, else only thread name.
        """

        self.__tmp_node_count = 0
        tname = self.unique_name if use_unique_thread_name else self.thread_name
        root = AggregatedStackFrame(StackFrame(tname, self.start_time, self.end_time))
        self._aggregate_all(root, self.stack_frames)
        return root

    def json(self) -> dict:
        if len(self.stack_frames) <= 0:
            return None

        k = f"{self.unique_name}[{self.start_time / 1_000_000:.3f},{self.end_time / 1_000_000:.3f}]"
        v = [c.json() for c in self.stack_frames]
        return {k: v}


class Perfdata:
    """Simple perf.data file parser."""

    def __init__(self,
                 record_file: StrPath,
                 symfs_dir: Optional[StrPath] = None,
                 kallsyms_file: Optional[StrPath] = None,
                 min_stack_count: int = 5) -> None:
        """Perfdata

        Args:
            record_file: perf.data file path, generated by simpelperf.
            symfs_dir: Symbols directory.
            kallsyms_file: Kernel symbol file path.
            min_stack_count: Due to some uncertain sample errors, there may be some wrong samples with shallow stack frames,
                use this value to filter such samples. When iterate sample, only samples which stack frames count greater than the value are yielded
        """

        self.record_file = Path(record_file)
        self.symfs_dir = Path(symfs_dir) if symfs_dir is not None else None
        self.kallsyms_file = Path(kallsyms_file) if kallsyms_file is not None else None
        self.min_stack_count = min_stack_count

    def iter_samples(self):
        """Iterate samples in record file."""

        lib = reportlib.GetReportLib(str(self.record_file))
        lib.ShowIpForUnknownSymbol()

        if self.symfs_dir is not None:
            lib.SetSymfs(str(self.symfs_dir))
        else:
            logger.warning("No symfs_dir given for record file %s", self.record_file)

        if self.kallsyms_file is not None:
            lib.SetKallsymsFile(str(self.kallsyms_file))

        supported_modes = lib.GetSupportedTraceOffCpuModes()
        logger.info("Supported modes: %s", supported_modes)

        if "on-off-cpu" in supported_modes:
            lib.SetTraceOffCpuMode("on-off-cpu")  # on-off-cpu  mixed-on-off-cpu

        count = 0
        while True:
            sample = lib.GetNextSample()
            count += 1

            if sample is None:
                lib.Close()
                break

            sample_info = SampleInfo(
                sample,
                lib.GetEventOfCurrentSample(),
                lib.GetSymbolOfCurrentSample(),
                lib.GetCallChainOfCurrentSample()
            )
            if sample_info.call_chain.num_entries >= self.min_stack_count:
                yield sample_info

            if count % 50000 == 0:
                logger.debug("%d samples iterated.", count)

    def get_threads(self) -> Dict[str, List[Thread]]:
        """Get threads from record file.

        Returns:
            threads: A dict maps thread_name to thread.
                There may be threads have the same thread_name, but different unique_name.
                Each list is sorted by thread samples count in descending order.
            """

        threads: Dict[str, Thread] = {}
        for sample_info in self.iter_samples():
            tname = sample_info.sample.thread_name
            tid = sample_info.sample.tid
            key = f"{tname}-{tid}"
            if key in threads:
                thread = threads[key]
            else:
                thread = threads[key] = Thread(tname, tid)

            thread.append_sample(sample_info)

        result: Dict[str, List[Thread]] = {}
        for thread in threads.values():
            logger.debug("Thread: %s, Sample count: %d", thread.unique_name, thread.samples_count)

            if thread.thread_name not in result:
                result[thread.thread_name] = []
            result[thread.thread_name].append(thread)

        for v in result.values():
            v.sort(key=lambda x: x.samples_count, reverse=True)

        return result
