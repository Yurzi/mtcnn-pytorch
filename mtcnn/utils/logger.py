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
    def __init__(self, log_writer: LogWriter) -> None:
        self.writer = log_writer

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
