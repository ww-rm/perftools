import os
from pathlib import Path
from typing import List, Optional, Union

import simpleperf_report_lib as reportlib

from .logger import logger

StrPath = Union[str, os.PathLike]


class Sample:
    def __init__(self, sample: reportlib.SampleStruct) -> None:
        self._sample = sample

    @property
    def ip(self) -> int:
        return self._sample.ip

    @property
    def pid(self) -> int:
        return self._sample.pid

    @property
    def tid(self) -> int:
        return self._sample.tid

    @property
    def thread_name(self) -> str:
        return self._sample.thread_comm

    @property
    def time(self) -> int:
        return self._sample.time

    @property
    def in_kernel(self) -> bool:
        return self._sample.in_kernel

    @property
    def cpu(self) -> int:
        return self._sample.cpu

    @property
    def period(self) -> int:
        return self._sample.period

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

    @property
    def name(self) -> str:
        return self._event.name

    def json(self) -> dict:
        return {
            "name": self.name
        }


class Symbol:
    def __init__(self, symbol: reportlib.SymbolStruct) -> None:
        self._symbol = symbol

    @property
    def dso_name(self) -> str:
        return self._symbol.dso_name

    @property
    def vaddr_in_file(self) -> int:
        return self._symbol.vaddr_in_file

    @property
    def symbol_name(self) -> str:
        return self._symbol.symbol_name

    @property
    def symbol_addr(self) -> int:
        return self._symbol.symbol_addr

    @property
    def symbol_len(self) -> int:
        return self._symbol.symbol_len

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

        self._symbol = Symbol(self._call_chain_entry.symbol)

    @property
    def ip(self) -> int:
        return self._call_chain_entry.ip

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    def json(self) -> dict:
        return {
            "ip": self.ip,
            "symbol": self.symbol.json()
        }


class CallChain:
    def __init__(self, call_chain: reportlib.CallChainStructure) -> None:
        self._call_chain = call_chain

        self._entries = [CallChainEntry(self._call_chain.entries[i]) for i in range(self._call_chain.nr)]

    def __len__(self) -> int:
        return self.num_entries

    @property
    def num_entries(self) -> int:
        return self._call_chain.nr

    @property
    def entries(self) -> List[CallChainEntry]:
        return self._entries

    def json(self) -> dict:
        return {
            "num_entries": self.num_entries,
            "entries": [v.json() for v in self.entries]
        }


class SampleInfo:
    """Aggregated sample information."""

    def __init__(self,
                 sample: reportlib.SampleStruct,
                 event: reportlib.EventStruct,
                 symbol: reportlib.SymbolStruct,
                 call_chain: reportlib.CallChainStructure) -> None:
        self.sample = Sample(sample)
        self.event = Event(event)
        self.symbol = Symbol(symbol)
        self.call_chain = CallChain(call_chain)

    def json(self) -> dict:
        return {
            "sample": self.sample.json(),
            "event": self.event.json(),
            "symbol": self.symbol.json(),
            "call_chain": self.call_chain.json()
        }


class StackFrameNode:
    def __init__(self, symbol: Symbol, begin_time: int, end_time: int) -> None:
        self.symbol = symbol
        self.begin_time = begin_time
        self.end_time = end_time

        self.sub_frame = []


class Thread:
    def __init__(self, name: str, tid: int) -> None:
        self.thread_name = name
        self.tid = tid
        self.unique_name = f"{name}-{tid}"

        self.stack_frames = []

    def _create_stack_frame(self, sample_info: SampleInfo) -> StackFrameNode:
        start_time = sample_info.sample.time - sample_info.sample.period
        end_time = sample_info.sample.time

        top_node = StackFrameNode(sample_info.symbol, start_time, end_time)

        node = top_node
        for entry in reversed(sample_info.call_chain.entries):
            node.sub_frame.append(StackFrameNode(entry.symbol, start_time, end_time))

        return top_node

    def append_sample(self, sample_info: SampleInfo):
        if len(self.stack_frames) <= 0:
            self.stack_frames.append(self._create_stack_frame(sample_info))
        else:
            ...


class Perfdata:
    """Simple perf.data file parser."""

    def __init__(self, record_file: StrPath, symfs_dir: Optional[StrPath] = None, kallsyms_file: Optional[StrPath] = None) -> None:
        self.record_file = Path(record_file)
        self.symfs_dir = Path(symfs_dir) if symfs_dir is not None else None
        self.kallsyms_file = Path(kallsyms_file) if kallsyms_file is not None else None

    def iter_samples(self):
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

        while True:
            sample = lib.GetNextSample()

            if sample is None:
                lib.Close()
                break

            yield SampleInfo(
                sample,
                lib.GetEventOfCurrentSample(),
                lib.GetSymbolOfCurrentSample(),
                lib.GetCallChainOfCurrentSample()
            )
