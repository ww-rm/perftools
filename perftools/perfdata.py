"""perf.data parser module."""

import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Union

import simpleperf_report_lib as reportlib

from . import stats_pb2
from .logger import logger

StrPath = Union[str, os.PathLike]


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

    @property
    def time_us(self) -> float:
        return self.time / 1000

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
    def start_time_us(self) -> float:
        return self.start_time / 1000

    @property
    def end_time_us(self) -> float:
        return self.end_time / 1000

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    @property
    def duration_us(self) -> float:
        return self.duration / 1000

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
    def min_time(self) -> int:
        return self.raw_stack_frames[0].start_time

    @property
    def max_time(self) -> int:
        return self.raw_stack_frames[-1].end_time

    @property
    def min_time_us(self) -> float:
        return self.min_time / 1000

    @property
    def max_time_us(self) -> float:
        return self.max_time / 1000

    @property
    def duration(self) -> int:
        return sum(sf.duration for sf in self.raw_stack_frames)

    @property
    def duration_us(self) -> float:
        return self.duration / 1000

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000

    @property
    def call_count(self) -> int:
        return len(self.raw_stack_frames)

    @property
    def avg_duration(self) -> float:
        return self.duration / self.call_count

    @property
    def avg_duration_us(self) -> float:
        return self.avg_duration / 1000

    def json(self) -> dict:
        if len(self.child_frames) <= 0:
            return None

        k = f"{self.symbol_name}[{self.duration / 1_000_000:.3f},{self.call_count}]"
        v = [c.json() for c in sorted(self.child_frames.values(), key=lambda x: x.avg_duration, reverse=True)]
        return {k: v}


class Timeline:
    """A container for stack frames (funtion calls)."""

    def __init__(self) -> None:
        self.stack_frames: List[StackFrame] = []
        # self.samples = []  # maybe can store samples
        self.samples_count = 0

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

    @property
    def start_time_us(self) -> float:
        return self.start_time / 1000

    @property
    def end_time_us(self) -> float:
        return self.end_time / 1000

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    @property
    def duration_us(self) -> float:
        return self.duration / 1000

    def _create_stack_frame(self, symbols: List[Symbol], start_time: int, end_time: int) -> StackFrame:
        """Create a new stack frame from symbols, the symbols do calls from first to last."""

        top_node = StackFrame(symbols[0].symbol_name, start_time, end_time)

        node = top_node
        for symbol in symbols[1:]:
            sub_node = StackFrame(symbol.symbol_name, start_time, end_time, node)
            node.child_frames.append(sub_node)
            node = sub_node

        return top_node

    def append_sample(self, sample_info: SampleInfo, max_stack_count: int = 30, start_time: Optional[int] = None):
        """Append a sample to the timeline tail."""

        max_stack_count = max(1, max_stack_count)

        self.samples_count += 1

        symbols = [sample_info.symbol]
        for s in sample_info.call_chain.entries:
            symbols.append(s.symbol)
        symbols.reverse()

        symbols = symbols[:max_stack_count]

        end_time = sample_info.sample.time

        if self.end_time > 0:
            start_time = self.end_time
        elif start_time is None:
            start_time = end_time - 1

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

        for agg_frame in agg_child_frames.values():
            child_frames = []
            for raw_frame in agg_frame.raw_stack_frames:
                child_frames.extend(raw_frame.child_frames)
            self._aggregate_all(agg_frame, child_frames)

    def aggregate(self, root_name: str) -> AggregatedStackFrame:
        """Aggregate all stack frames to a root AggregatedStackFrame, the timeline will be seen as a function call.

        Args:
            root_name: The root name of AggregatedStackFrame.
        """

        root = AggregatedStackFrame(StackFrame(root_name, self.start_time, self.end_time))
        self._aggregate_all(root, self.stack_frames)
        return root


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

        self.timelines: List[Timeline] = []

    @property
    def start_time(self) -> int:
        """Returns start time of first top frame, returns -1 if no frames."""

        if len(self.timelines) <= 0:
            return -1
        return self.timelines[0].start_time

    @property
    def end_time(self) -> int:
        """Returns end time of last top frame, returns -1 if no frames."""

        if len(self.timelines) <= 0:
            return -1
        return self.timelines[-1].end_time

    @property
    def start_time_us(self) -> float:
        return self.start_time / 1000

    @property
    def end_time_us(self) -> float:
        return self.end_time / 1000

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    @property
    def duration_us(self) -> float:
        return self.duration / 1000

    def append_sample(self, sample_info: SampleInfo, frame_end: int, max_stack_count: int = 30):
        ...

    def aggregate(self, use_unique_thread_name: bool = False) -> AggregatedStackFrame:
        """Aggregate all stack frames to a root AggregatedStackFrame, the thread will be seen as a function call.

        Args:
            use_unique_thread_name: If true, set root thread name with thread id, else only thread name.
        """

        tname = self.unique_name if use_unique_thread_name else self.thread_name
        return super().aggregate(tname)


class Perfdata:
    """Simple perf.data file parser."""

    @staticmethod
    def read_frame_timestamps(filepath: StrPath) -> List[int]:
        timestamps = []
        with open(filepath) as f:
            for line in f:
                timestamps.append(int(line.strip().split()[0]))

        if len(timestamps) <= 2:
            return ValueError(f"Length of frame_timestamps must greater than 2, given {timestamps}")
        return timestamps

    def __init__(self,
                 record_file: StrPath,
                 frame_timestamps_file: StrPath,
                 symfs_dir: Optional[StrPath] = None,
                 kallsyms_file: Optional[StrPath] = None,
                 min_stack_count: int = 5,
                 max_stack_count: int = 30) -> None:
        """Perfdata

        Args:
            record_file: perf.data file path, generated by simpelperf.
            symfs_dir: Symbols directory.
            kallsyms_file: Kernel symbol file path.
            min_stack_count: Due to some uncertain sample errors, there may be some wrong samples with shallow stack frames,
                use this value to filter such samples. When iterate sample, only samples which stack frames count greater than the value are yielded
            max_stack_count: Due to limitation of pb files (must smaller than 64 MB), so we limit the depth of stacks of samples here when we do aggregating.
        """

        self.record_file = Path(record_file)
        self.frame_timestamps = self.read_frame_timestamps(frame_timestamps_file)
        self.symfs_dir = Path(symfs_dir) if symfs_dir is not None else None
        self.kallsyms_file = Path(kallsyms_file) if kallsyms_file is not None else None
        self.min_stack_count = min_stack_count
        self.max_stack_count = max_stack_count
        self.important_thread_prefix = [
            "GameThread",
            "RenderThread",
            "RHIThread",
            "UnityMain"
        ]

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
        last_time = time.time()
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

            count += 1
            now_time = time.time()
            if now_time - last_time > 10:
                last_time = now_time
                logger.debug("%d samples iterated.", count)

            # filter shallow sample
            if sample_info.call_chain.num_entries < self.min_stack_count:
                continue

            # filter threads we do not care
            if not any(sample_info.sample.thread_name.startswith(v) for v in self.important_thread_prefix):
                continue

            yield sample_info

        logger.debug("All %d samples iterate done.", count)

    def get_threads(self) -> Dict[str, List[Thread]]:
        """Get threads from record file.

        Returns:
            threads: A dict maps thread_name to thread.
                There may be threads have the same thread_name, but different unique_name.
                Each list is sorted by thread samples count in descending order.
        """

        samples = list(self.iter_samples())
        samples.sort(key=lambda v: v.sample.time)

        threads: Dict[str, Thread] = {}
        frame_idx = 1
        sample_idx = 0
        while frame_idx < len(self.frame_timestamps) and sample_idx < len(samples):
            frame_start = self.frame_timestamps[frame_idx - 1]
            frame_end = self.frame_timestamps[frame_idx]
            sample_info = samples[sample_idx]

            # [ frame - 1 ) [ frame ) [ frame + 1 )
            #       ^
            if sample_info.sample.time_us < frame_start:
                sample_idx += 1
                continue

            # [ frame - 1 ) [ frame ) [ frame + 1 )
            #                            ^
            if sample_info.sample.time_us >= frame_end:
                frame_idx += 1
                continue

            tname = sample_info.sample.thread_name
            tid = sample_info.sample.tid
            key = f"{tname}-{tid}"
            if key in threads:
                thread = threads[key]
            else:
                thread = threads[key] = Thread(tname, tid)

            if len(thread.timelines) <= 0:
                last_timeline = Timeline()
                thread.timelines.append(last_timeline)
            else:
                last_timeline = thread.timelines[-1]
                if last_timeline.end_time_us < frame_end:

                thread.timelines.append(last_timeline)
            else

        result: Dict[str, List[Thread]] = {}
        for thread in threads.values():
            logger.debug("Thread: %s, Sample count: %d", thread.unique_name, thread.samples_count)

            if thread.thread_name not in result:
                result[thread.thread_name] = []
            result[thread.thread_name].append(thread)

        for v in result.values():
            v.sort(key=lambda x: x.samples_count, reverse=True)

        return result

    def get_stats_pb_message(self, frame_timestamps: Optional[List[int]] = None, fake_root_name: str = "ThreadRoot") -> stats_pb2.AggregatedStatStackNode:
        """Get AggregatedStatStackNode of all threads.
            For threads has same name, only keep the thread has most samples.
        """

        threads = self.get_threads()

        fake_root = stats_pb2.AggregatedStatStackNode()
        fake_root.meta.event_name = fake_root_name

        for v in threads.values():
            important_thread = v[0]
            logger.debug("Keep thread: %s", important_thread.unique_name)
            fake_root.children.append(important_thread.aggregate().get_stats_pb_message(frame_timestamps))

        return fake_root

    def write_to_pa_file(self, path: StrPath, timestamps_path: Optional[StrPath] = None):
        """Write perfdata to pa file. Just a stats_pb2.PreciseAnalysisResult object."""

        if timestamps_path is None:
            frame_timestamps = None
        else:
            frame_timestamps = self.read_frame_timestamps(timestamps_path)

        agg_node = self.get_stats_pb_message(frame_timestamps)
        logger.debug("Total %d threads kept", len(agg_node.children))

        precise_result = stats_pb2.PreciseAnalysisResult()
        precise_result.aggregated_stat_stack_node.CopyFrom(agg_node)
        with open(path, "wb") as f:
            f.write(precise_result.SerializeToString())


def main():
    from datetime import datetime
    from .utils import upload_pa_file
    parser = ArgumentParser()
    parser.add_argument("-i", "--record-file", default="perf.data", help="Simpleperf record file.")
    parser.add_argument("-o", "--output-dir", default="", help="output directory of pa file, default to current dir.")

    parser.add_argument("-s", "--symfs", help="Set the path to find binaries with symbols and debug info.")
    parser.add_argument("-k", "--kallsyms", help="Set the path to find kernel symbols.")
    parser.add_argument("-t", "--frame-timestamps", help="Game frames timestamps file.")

    parser.add_argument("--min-stack-count", type=int, default=5)

    parser.add_argument("--pid", default="pid", help="projec id, such as LK or UC.")
    parser.add_argument("--map", default="map", help="map name, such as Bigworld01 or farmland.")
    parser.add_argument("--scene", default="scene", help="scene name, such as 10人综合 or any other identifier.")
    parser.add_argument("--mobile", default="mobile", help="mobile, such as iphone14pro. There should be no spaces in mobile name.")
    parser.add_argument("--quality", default="quality", help="quality, such as 高, 默认")
    parser.add_argument("--version", default="version_num", help="such as 0.1.3_1")

    parser.add_argument("--username", help="性能平台API用户名")
    parser.add_argument("--usertoken", help="性能平台API用户令牌")
    parser.add_argument("--upload-to-prod", action="store_true", help="上传至正式环境, 否则默认测试环境")

    args = parser.parse_args()

    pa_filename = "-".join([
        args.pid,
        args.map,
        args.scene,
        args.mobile,
        args.quality,
        args.version,
        datetime.now().strftime("%Y%m%d%H%M%S"),
        "Client"
    ]) + ".pa"

    pa_filepath = Path(args.output_dir, pa_filename)

    perfdata = Perfdata(args.record_file, args.symfs, args.kallsyms, args.min_stack_count)
    perfdata.write_to_pa_file(pa_filepath, args.frame_timestamps)

    if args.username and args.usertoken:
        upload_pa_file(args.pid, args.username, args.usertoken, pa_filepath, args.upload_to_prod)


if __name__ == "__main__":
    main()
