"""perf.data parser module."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import simpleperf_report_lib as reportlib

from .logger import logger

StrPath = Union[str, os.PathLike]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Simpleperf Sample Data Structures >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class Sample:
    def __init__(self, sample: reportlib.SampleStruct) -> None:
        self.ip: int = sample.ip
        self.pid: int = sample.pid
        self.tid: int = sample.tid
        self.thread_name: str = sample.thread_comm
        self.time: int = sample.time
        self.in_kernel: bool = sample.in_kernel
        self.cpu: int = sample.cpu
        self.period: int = sample.period

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
        self.name: str = event.name

    def json(self) -> dict:
        return {
            "name": self.name
        }


class Symbol:
    def __init__(self, symbol: reportlib.SymbolStruct) -> None:
        self.dso_name: str = symbol.dso_name
        self.vaddr_in_file: int = symbol.vaddr_in_file
        self.symbol_name: str = symbol.symbol_name
        self.symbol_addr: int = symbol.symbol_addr
        self.symbol_len: int = symbol.symbol_len

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
        self.ip: int = call_chain_entry.ip
        self.symbol = Symbol(call_chain_entry.symbol)

    def json(self) -> dict:
        return {
            "ip": self.ip,
            "symbol": self.symbol.json()
        }


class CallChain:
    def __init__(self, call_chain: reportlib.CallChainStructure) -> None:
        self.num_entries: int = call_chain.nr
        self.entries = [CallChainEntry(call_chain.entries[i]) for i in range(call_chain.nr)]

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

    def __repr__(self) -> str:
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

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Simpleperf Sample Data Structures <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Frame Data Structures >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class FrameBase:
    """Frame base class.
        Include left point and exclude right point, 
        which means the same behavior as range, [start_time, end_time)

    The unit of time is nanosecond (ns, 10^-9 second)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @property
    def start_time(self) -> int:
        raise NotImplementedError

    @property
    def end_time(self) -> int:
        raise NotImplementedError

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    @property
    def start_time_us(self) -> float:
        return self.start_time / 1000

    @property
    def end_time_us(self) -> float:
        return self.end_time / 1000

    @property
    def duration_us(self) -> float:
        return self.duration / 1000

    @property
    def start_time_ms(self) -> float:
        return self.start_time_us / 1000

    @property
    def end_time_ms(self) -> float:
        return self.end_time_us / 1000

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000


class FrameContainerBase(FrameBase):
    """Frame container base, consists of continious frames."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.frames: List[FrameBase] = []

    @property
    def is_empty(self) -> bool:
        return len(self.frames) <= 0

    @property
    def start_time(self) -> int:
        """Returns start time of first top frame, returns -1 if no frames."""

        if len(self.frames) <= 0:
            return -1
        return self.frames[0].start_time

    @property
    def end_time(self) -> int:
        """Returns end time of last top frame, returns -1 if no frames."""

        if len(self.frames) <= 0:
            return -1
        return self.frames[-1].end_time


class StackFrame(FrameBase):
    """Stack frame used in threads.

    Each stack frame is a Bidirectional Multi-way Tree.
        Use father_frame to get father node, and child_frames to get children nodes.

    ```plain
    +----------------------------------------------+
    |         Father Frame (Optional)              |
    +--+-----------------------------------------+-+
       |                Frame (self)             |
       +---------------+------------+------------+
       |    Sub Frame  |            | Sub Frame  |
       +--------+------+            +------------+
       ^        | Sub  |                         ^
       ^        +------+                         ^
       ^                                         ^
    start_time                                end_time
    ```
    """

    @staticmethod
    def create(symbols: List[Symbol], start_time: int, end_time: int, father_frame: Optional["StackFrame"] = None):
        """Create stack frames with symbols and times.

        ```plain
             +----------------+
             |  Father Frame  |
             +----------------+
             |  Frame (self)  |
             +----------------+
             |    Sub Frame   |
             +----------------+
             |       ...      |
             +----------------+
             |    Sub Frame   |
             +----------------+
             ^                ^
        start_time         end_time
        ```
        """

        if len(symbols) <= 0:
            raise ValueError("Empty symbols")

        top_node = StackFrame(symbols[0].symbol_name, start_time, end_time, father_frame)

        node = top_node
        for symbol in symbols[1:]:
            sub_node = StackFrame(symbol.symbol_name, start_time, end_time, node)
            node.child_frames.append(sub_node)
            node = sub_node

        return top_node

    def __init__(self, symbol_name: str, start_time: int, end_time: int, father_frame: Optional["StackFrame"] = None) -> None:
        """Create a stack frame without sub calls."""

        super().__init__()

        self.symbol_name = symbol_name
        self._start_time = start_time
        self._end_time = end_time
        self.father_frame = father_frame

        self.child_frames: List[StackFrame] = []

    @property
    def start_time(self) -> int:
        return self._start_time

    @start_time.setter
    def start_time(self, value: int):
        self._start_time = value

    @property
    def end_time(self) -> int:
        return self._end_time

    @end_time.setter
    def end_time(self, value: int):
        self._end_time = value

    def extend(self, symbols: List[Symbol], sample_time: int):
        """Extend the frame tail with given symbols, will try to merge from top to bottom if possible.
            S1 calls S2 calls S3 ... calls Sn.

        Raises:
            ValueError: Different top node symbol name.
        """

        if len(symbols) <= 0:
            return

        if sample_time <= self.end_time:
            return

        if self.symbol_name != symbols[0].symbol_name:
            raise ValueError("Can't extend to different top node.")

        top_end_time = self.end_time
        current_stack_frame = self
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
                current_stack_frame.end_time = sample_time

                # continue to compare sub stack frames
                if len(current_stack_frame.child_frames) > 0:
                    current_stack_frame = current_stack_frame.child_frames[-1]

                # current stack frames count less than the sample
                # make the rest symbols a sub frame to current stack frame
                elif i + 1 <= len(symbols) - 1:
                    sub_stack_frame = self.create(symbols[i+1:], top_end_time, sample_time, current_stack_frame)
                    current_stack_frame.child_frames.append(sub_stack_frame)
                    return

            # a new sub frame to current father frame
            else:
                father_frame = current_stack_frame.father_frame
                assert father_frame is not None

                sub_stack_frame = self.create(symbols[i:],
                                              current_stack_frame.end_time,  # maybe we can use top_end_time ?
                                              sample_time,
                                              father_frame)

                father_frame.child_frames.append(sub_stack_frame)

                return

    def json(self) -> dict:
        k = f"{self.symbol_name}[{self.start_time_ms:.3f},{self.end_time_ms:.3f}]"

        if len(self.child_frames) <= 0:
            v = None
        else:
            v = [c.json() for c in self.child_frames]
        return {k: v}


class AggregatedStackFrame(FrameBase):
    """Aggregated stack frame for threads.

    Each stack frame is a Bidirectional Multi-way Tree.
        Use father_frame to get father node, and child_frames to get children nodes.

    raw_stack_frames stores all aggregated raw stack frames.

    The unit of time is nanosecond (ns, 10^-9 second)
    """

    @staticmethod
    def _aggregate_child_frames(root: "AggregatedStackFrame"):
        child_frames = [child for frame in root.raw_stack_frames for child in frame.child_frames]

        for frame in child_frames:
            if frame.symbol_name not in root.child_frames:
                root.child_frames[frame.symbol_name] = AggregatedStackFrame(frame, root)
            else:
                root.child_frames[frame.symbol_name].raw_stack_frames.append(frame)

        for agg_frame in root.child_frames.values():
            AggregatedStackFrame._aggregate_child_frames(agg_frame)

    @staticmethod
    def create(stack_frames: List[StackFrame]) -> Dict[str, "AggregatedStackFrame"]:
        """Create AggregatedStackFrame from some root stack frames."""

        result: Dict[str, "AggregatedStackFrame"] = {}

        for frame in stack_frames:
            if frame.symbol_name not in result:
                result[frame.symbol_name] = AggregatedStackFrame(frame)  # unique root node
            else:
                result[frame.symbol_name].raw_stack_frames.append(frame)

        for agg_frame in result.values():
            AggregatedStackFrame._aggregate_child_frames(agg_frame)

        return result

    def __init__(self, stack_frame: StackFrame, father_frame: Optional["AggregatedStackFrame"] = None) -> None:
        """Do not use it directly, use static method create instead."""

        self.symbol_name = stack_frame.symbol_name
        self.raw_stack_frames: List[StackFrame] = [stack_frame]
        self.father_frame = father_frame
        self.child_frames: Dict[str, AggregatedStackFrame] = {}

    @property
    def start_time(self) -> int:
        return self.raw_stack_frames[0].start_time

    @property
    def end_time(self) -> int:
        return self.raw_stack_frames[-1].end_time

    @property
    def duration(self) -> int:
        return sum(v.duration for v in self.raw_stack_frames)

    @property
    def call_count(self) -> int:
        return len(self.raw_stack_frames)

    def json(self) -> dict:
        k = f"{self.symbol_name}[{self.duration_ms:.3f},{self.call_count}]"

        if len(self.child_frames) <= 0:
            v = None
        else:
            v = [c.json() for c in sorted(self.child_frames.values(), key=lambda x: x.duration, reverse=True)]
        return {k: v}


class TimeFrame(FrameContainerBase):
    """A time based container for stack frames (funtion calls).

    A time frame consists of continous stack frames, and each stack frames was created by many samples.

    ```plain
    +----------------------------------------------+
    |                 Time Frame                   |
    +-------------+------------------+-------------+
    | Stack Frame |   Stack Frame    | Stack Frame |
    +^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^+
    |                   Samples                    |
    +----------------------------------------------+
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.frames: List[StackFrame]

    def extend(self, end_time: int):
        """Extend the last call stack to end_time.

        ```plain
            +------------------------------+~~~~~~~~~~+
            |          Frame               ~          |
            +------------------------------+~~~~~~~~~~+
            |         Sub Frame            ~          |
            +---------------+--------------+~~~~~~~~~~+
            |    Sub Frame  |    | Sub |   ^          ^
            +--------+------+    +-----+   ^          ^
            ^        | Sub  |              ^          ^
            ^        +------+              ^          ^
            ^                              ^          ^
        start_time                    end_time' -> end_time
        ```
        """

        if len(self.frames) <= 0:
            return

        if end_time <= self.end_time:
            return

        top_end_time = self.frames[-1].end_time
        node = self.frames[-1]
        while True:
            if node.end_time < top_end_time:
                break
            node.end_time = end_time

            if len(node.child_frames) <= 0:
                break
            node = node.child_frames[-1]

    def append(self, symbols: List[Symbol], sample_time: int, start_time: Optional[int] = None):
        """Append a sample to the frame tail.

        Args:
            symbols: Symbols of sample.
            sample_time: Timestamp of sample, nanosecond.
            start_time: Only used when self is newly created.
        """

        if sample_time <= self.end_time:
            return

        if start_time and start_time <= 0:
            start_time = None

        # a newly created time frame
        if len(self.frames) <= 0:
            self.frames.append(StackFrame.create(symbols, (start_time or sample_time), sample_time))
            return

        last_stack_frame = self.frames[-1]
        if symbols[0].symbol_name == last_stack_frame.symbol_name:
            # try merge stack frames
            last_stack_frame.extend(symbols, sample_time)
        else:
            # a new root stack frame
            self.frames.append(StackFrame.create(symbols, self.end_time, sample_time))

    def aggregate(self) -> Dict[str, AggregatedStackFrame]:
        """Aggregate all stack frames to AggregatedStackFrame."""

        return AggregatedStackFrame.create(self.frames)


class Thread(FrameContainerBase):
    """Thread, a container for time frames.

    ```plain
    +----------------------------------------------+
    |                   Thread                     |
    +----------------------------------------------+
    |    Time Frame    |        Time Frame         |
    +^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^+
    |                   Samples                    |
    +----------------------------------------------+
    ```
    """

    def __init__(self, name: str, tid: int, max_stack_count: int = 30, exclude_kernel_symbol: bool = True) -> None:
        """Thread, a container for time frames.

        Args:
            name: Thread name, such as UnityMain or GameThread
            tid: Thread id, sampled by simpleperf.
            max_stack_count: Max stack count when appending samples.
            exclude_kernel_symbol: Strip kallsyms symbols from tail of call chain.
        """

        super().__init__()

        self.thread_name = name
        self.tid = tid
        self.unique_name = f"{name}-{tid}"
        self.max_stack_count = max_stack_count
        self.exclude_kernel_symbol = exclude_kernel_symbol

        self.frames: List[TimeFrame]
        self._next_need_new_frame: bool = True
        self.samples_count = 0

    def __repr__(self) -> str:
        values = ", ".join([
            f"name={self.thread_name}",
            f"tid={self.tid}",
            f"duration={self.duration_ms:.6f}",
            f"frames_count={len(self.frames)}",
            f"samples_count={self.samples_count}"
        ])
        return f"Thread({values})"

    def finish_current_frame(self, frame_end: int):
        """Finish current time frame, and next sample will be append to a new time frame."""

        if len(self.frames) > 0:
            self.frames[-1].extend(frame_end)
        self._next_need_new_frame = True

    def append(self, sample_info: SampleInfo):
        """Append a sample.

        Use finish_current_frame to control time frame range.
        """

        sample_time = sample_info.sample.time
        if sample_time <= self.end_time:
            logger.warning("skip %s", sample_info)
            return

        symbols = [sample_info.symbol]
        for s in sample_info.call_chain.entries:
            symbols.append(s.symbol)
        symbols.reverse()

        # limit the stack depth
        symbols = symbols[:self.max_stack_count]

        # strip kernel symbols in tail
        if self.exclude_kernel_symbol:
            while len(symbols) > 0 and "kernel.kallsyms" in symbols[-1].symbol_name:
                symbols.pop()

        if len(symbols) <= 0:
            return

        # a newly created thread or current frame need to be finished
        # we need to add an empty time frame firstly
        if len(self.frames) <= 0 or self._next_need_new_frame:
            self.frames.append(TimeFrame())

        if len(self.frames) <= 1:
            self.frames[-1].append(symbols, sample_time)
        else:
            # use end_time to ensure the frames are continous
            self.frames[-1].append(symbols, sample_time, self.frames[-2].end_time)

        self.samples_count += 1
        self._next_need_new_frame = False

    def aggregate_frames(self) -> List[AggregatedStackFrame]:
        """Aggregate all time frames to some AggregatedStackFrame,
            each aggregated time frame has a fake root function call with thread name.
        """

        agg_frames = []
        for frame in self.frames:
            agg_frame = AggregatedStackFrame(StackFrame(self.thread_name, frame.start_time, frame.end_time))
            agg_frame.child_frames.update(frame.aggregate())
            agg_frames.append(agg_frame)

        return agg_frames

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Frame Data Structures <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class Perfdata:
    """Simple perf.data file parser."""

    @staticmethod
    def read_frame_timestamps(filepath: StrPath) -> List[int]:
        """Read frame timestamps from plain txt file, only read first column.

        The time unit of file is microsecond (us, 10^-6 second)

        Raises:
            ValueError: Timestamps is too few.
        """

        timestamps = []
        with open(filepath) as f:
            for line in f:
                timestamps.append(int(line.strip().split()[0]) * 1000)  # convert to nanosecond

        if len(timestamps) <= 10:
            return ValueError(f"Too few frame_timestamps, given {len(timestamps)}")
        return timestamps

    def __init__(
        self,
        record_file: StrPath,
        frame_timestamps_file: StrPath,
        symfs_dir: Optional[StrPath] = None,
        kallsyms_file: Optional[StrPath] = None,
        min_stack_count: int = 5,
        max_stack_count: int = 30,
        exclude_kernel_symbol: bool = True,
        important_thread_prefix=("GameThread", "RenderThread", "RHIThread", "UnityMain"),
        *,
        cache_samples: bool = False
    ) -> None:
        """Perfdata

        Args:
            record_file: perf.data file path, generated by simpelperf.
            frame_timestamps_file: Frame timestamps file path.
            symfs_dir: Symbols directory.
            kallsyms_file: Kernel symbol file path.
            min_stack_count: Due to some uncertain sample errors, there may be some wrong samples with shallow stack frames,
                use this value to filter such samples. When iterate sample, only samples which stack frames count greater than the value are yielded
            max_stack_count: Due to limitation of pb files (must smaller than 64 MB), we may limit the depth of stacks of each thread when we do aggregating.
            exclude_kernel_symbol: Strip kallsyms symbols from tail of call chain.
            important_thread_prefix: string prefix filters for thread name.

        Keyword Args:
            cache_samples: Whether cache samples read from record file.
        """

        self.record_file = Path(record_file)
        self.frame_timestamps = self.read_frame_timestamps(frame_timestamps_file)
        self.symfs_dir = Path(symfs_dir) if symfs_dir is not None else None
        self.kallsyms_file = Path(kallsyms_file) if kallsyms_file is not None else None
        self.min_stack_count = min_stack_count
        self.max_stack_count = max_stack_count
        self.exclude_kernel_symbol = exclude_kernel_symbol
        self.important_thread_prefix = important_thread_prefix
        self._samples: List[SampleInfo] = None
        self.cache_samples = cache_samples

    @property
    def samples(self) -> List[SampleInfo]:
        if self._samples is not None:
            return self._samples

        samples = sorted(self.iter_samples(), key=lambda x: x.sample.time)
        logger.info("Filtered samples count: %d", len(samples))
        if self.cache_samples:
            self._samples = samples
        return samples

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

        __count = 0
        __last_time = time.time()
        while True:
            sample = lib.GetNextSample()

            if sample is None:
                lib.Close()
                break

            sample_info = SampleInfo(
                sample,
                lib.GetEventOfCurrentSample(),
                lib.GetSymbolOfCurrentSample(),
                lib.GetCallChainOfCurrentSample()
            )

            __count += 1
            __now_time = time.time()
            if __now_time - __last_time > 10:
                __last_time = __now_time
                logger.debug("%d samples iterated.", __count)

            # filter shallow sample
            if sample_info.call_chain.num_entries < self.min_stack_count:
                continue

            # filter threads we do not care
            if not any(sample_info.sample.thread_name.startswith(v) for v in self.important_thread_prefix):
                continue

            yield sample_info

        logger.info("%d samples iterated totally.", __count)

    def get_threads(self) -> Dict[str, List[Thread]]:
        """Get threads from record file.

        Returns:
            threads: A dict maps thread_name to thread.
                There may be threads have the same thread_name, but different unique_name.
                Each list is sorted by thread samples count in descending order.

        Raises:
            ValueError: Invalid frame timestamps to samples.
        """

        samples = self.samples

        threads: Dict[str, Thread] = {}

        if self.frame_timestamps[0] >= samples[-1].sample.time or self.frame_timestamps[-1] <= samples[0].sample.time:
            raise ValueError("frame timestamps no intersection with samples time")

        frame_idx = 1
        sample_idx = 0

        # find first position where a sample in a frame
        while frame_idx < len(self.frame_timestamps) and sample_idx < len(samples):
            frame_start = self.frame_timestamps[frame_idx - 1]
            frame_end = self.frame_timestamps[frame_idx]
            sample_info = samples[sample_idx]

            # [ frame - 1 ) [ frame ) [ frame + 1 )
            #       ^
            if sample_info.sample.time < frame_start:
                sample_idx += 1
                continue

            # [ frame - 1 ) [ frame ) [ frame + 1 )
            #                            ^
            if sample_info.sample.time >= frame_end:
                frame_idx += 1
                continue

            break

        # iterate samples in each frame
        __count = 0
        __last_time = time.time()
        while frame_idx < len(self.frame_timestamps) and sample_idx < len(samples):
            frame_start = self.frame_timestamps[frame_idx - 1]
            frame_end = self.frame_timestamps[frame_idx]
            sample_info = samples[sample_idx]

            # here we should goto next frame
            if sample_info.sample.time >= frame_end:
                for t in threads.values():
                    t.finish_current_frame(frame_end)
                frame_idx += 1
                continue

            assert sample_info.sample.time >= frame_start

            tname = sample_info.sample.thread_name
            tid = sample_info.sample.tid
            key = f"{tname}-{tid}"
            if key in threads:
                thread = threads[key]
            else:
                thread = threads[key] = Thread(tname, tid, self.max_stack_count, self.exclude_kernel_symbol)

            thread.append(sample_info)
            sample_idx += 1

            __count += 1
            __now_time = time.time()
            if __now_time - __last_time > 10:
                __last_time = __now_time
                logger.debug("%d samples processed.", __count)

        logger.info("%d samples processed totally.", __count)

        result: Dict[str, List[Thread]] = {}
        for thread in threads.values():
            logger.debug("%s", thread)

            if thread.thread_name not in result:
                result[thread.thread_name] = []
            result[thread.thread_name].append(thread)

        for v in result.values():
            v.sort(key=lambda x: x.samples_count, reverse=True)

        return result
