from mtcnn.utils.logger import ConsoleLogWriter, DebugLogger

logger = DebugLogger(__name__, ConsoleLogWriter())

if __name__ == "__main__":
    logger({"msg": "test console logger"})
    logger({"qwq"})

    logger("qwq", {"awa"}, {"qwq": "qwq"}, ["awa", {"qwq": "qwq"}])
    logger.info("test line number")
