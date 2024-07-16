import os
from pathlib import Path
from typing import Dict, List, Optional, Union

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

        self.start_time = self.sample.time - self.sample.period
        self.end_time = self.sample.time

    def __str__(self) -> str:
        values = ", ".join([
            f"thread={self.sample.thread_name}-{self.sample.tid}",
            f"symbol={self.symbol.symbol_name}",
            f"period=({self.start_time},{self.end_time})"
        ])
        return f"SampleInfo({values})"

    def json(self) -> dict:
        return {
            "sample": self.sample.json(),
            "event": self.event.json(),
            "symbol": self.symbol.json(),
            "call_chain": self.call_chain.json()
        }


class AggregatedCallNode:
    def __init__(self, symbol: Symbol, begin_time: int, end_time: int) -> None:
        self.symbol = symbol
        self.begin_time = begin_time
        self.end_time = end_time

        self.nodes: Dict[str, AggregatedCallNode] = {}  # symbol_name -> node

    @property
    def duration(self) -> int:
        return self.end_time - self.begin_time


class AggregatedThread:
    def __init__(self, name: str, tid: int) -> None:
        self.thread_name = name
        self.tid = tid
        self.unique_name = f"{name}-{tid}"

        self.call_nodes: Dict[str, AggregatedCallNode] = {}  # symbol_name -> node

    def _create_callnode(self, sample_info: SampleInfo) -> AggregatedCallNode:
        top_node = AggregatedCallNode(sample_info.symbol, sample_info.start_time, sample_info.end_time)

        for entry in sample_info.call_chain.entries:
            node = AggregatedCallNode(entry.symbol, sample_info.start_time, sample_info.end_time)
            node.nodes.append(top_node)
            top_node = node

        return top_node

    def _aggregate_callnode(self, sample_info: SampleInfo) -> AggregatedCallNode:
        ...

    def append_sample(self, sample_info: SampleInfo):
        if len(self.call_nodes) <= 0:
            self.call_nodes.append(self._create_callnode(sample_info))
        else:
            last_node = self.call_nodes[-1]
            if sample_info.end_time <= last_node.end_time:
                logger.warning("skip sample %s", sample_info)
                return


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
