import inspect
from typing import Any, Dict, List, Set


class LogItem:
    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value

    @staticmethod
    def from_dict(input: Dict) -> list["LogItem"]:
        res: list[LogItem] = list()

        for key, value in input.items():
            res.append(LogItem(key, value))
        return res

    @staticmethod
    def from_set(input: Set) -> list["LogItem"]:
        res: list[LogItem] = list()

        for item in input:
            res.append(LogItem("", item))
        return res

    @staticmethod
    def from_list(input: List) -> list["LogItem"]:
        res: list[LogItem] = list()

        for item in input:
            res.append(LogItem("", item))
        return res


class LogWriter:
    def __call__(self, log_items: list[LogItem]) -> None:
        raise NotImplementedError


class Logger:
    def __init__(self, module_name:str, log_writer: LogWriter, level:int = 1) -> None:
        self.writer = log_writer
        self.level = level
        self.module_name = module_name

    def log(self, input: list[LogItem]) -> None:
        self.writer(input)

    def __call__(self, *log_items: Dict | Set | List | str) -> None:
        items: list[LogItem] = list()

        for item in log_items:
            if isinstance(item, dict):
                items.extend(LogItem.from_dict(item))
            elif isinstance(item, set):
                items.extend(LogItem.from_set(item))
            elif isinstance(item, list):
                items.extend(LogItem.from_list(item))
            elif isinstance(item, str):
                items.append(LogItem("", item))
            else:
                raise TypeError(f"unsupported type {type(item)}")

        self.log(items)
    def debug(self, msg: str):
        print("NotImplementedError: This is a abstract method\n", msg)

    def info(self, msg: str):
        print("NotImplementedError: This is a abstract method\n", msg)

    def warn(self, msg: str):
        print("NotImplementedError: This is a abstract method\n", msg)

    def error(self, msg: str):
        print("NotImplementedError: This is a abstract method\n", msg)

    def fatal(self, msg: str):
        print("NotImplementedError: This is a abstract method\n", msg)

    @staticmethod
    def get_lineno() -> int:
        lineno = -1
        c_frame = inspect.currentframe()

        # check frame
        if c_frame is None:
            return lineno
        lineno = c_frame.f_lineno 
        outer_frame = c_frame.f_back
        if outer_frame is None:
            return lineno
        lineno = outer_frame.f_lineno
        outer_frame = outer_frame.f_back
        if outer_frame is None:
            return lineno
        lineno = outer_frame.f_lineno
        return lineno


class ConsoleLogWriter(LogWriter):
    def __call__(self, log_items: list[LogItem]) -> None:
        for item in log_items:
            if len(item.key) == 0 and len(item.value) == 0:
                continue

            if len(item.key) == 0:
                print(f"{item.value}", end=" ")
            elif len(item.value) == 0:
                print(f"[{item.key}]", end=" ")
            else:
                print(f"[{item.key}]: {item.value}", end=" ")

        # end of one line
        print()

from tqdm import tqdm

class TqdmLogWriter(LogWriter):
    def __call__(self, log_items: list[LogItem]) -> None:
        for item in log_items:
            if len(item.key) == 0 and len(item.value) == 0:
                continue;

            if len(item.key) == 0:
                tqdm.write(f"{item.value}", end=" ")
            elif len(item.value) == 0:
                tqdm.write(f"[{item.key}]", end=" ")
            else:
                tqdm.write(f"[{item.key}]: {item.value}", end=" ")
                
        # end of one line
        tqdm.write("")



class DebugLogger(Logger):
    def __init__(self, module_name: str, log_writer: LogWriter, level: int = 1) -> None:
        super(DebugLogger, self).__init__(module_name, log_writer, level)

    def debug(self, msg: str):
        if self.level <= 0:
            lineno = self.get_lineno()
            self(f"[{self.module_name}:{lineno}]", {"DEBUG": msg})

    def info(self, msg: str):
        if self.level <= 1:
            lineno = self.get_lineno()
            self(f"[{self.module_name}:{lineno}]", {"INFO": msg})

    def warn(self, msg: str):
        if self.level <= 2:
            lineno = self.get_lineno()
            self(f"[{self.module_name}:{lineno}]", {"WARN": msg})

    def error(self, msg: str):
        if self.level <= 3:
            lineno = self.get_lineno()
            self(f"[{self.module_name}:{lineno}]", {"ERROR": msg})

    def fatal(self, msg: str):
        if self.level <= 4:
            lineno = self.get_lineno()
            self(f"[{self.module_name}:{lineno}]", {"FATAL": msg})

