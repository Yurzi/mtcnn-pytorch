from mtcnn.utils.logger import ConsoleLogWriter, Logger

logger = Logger(ConsoleLogWriter())

if __name__ == "__main__":
    logger({"msg": "test console logger"})
    logger({"qwq"})

    logger("qwq", {"awa"}, {"qwq": "qwq"}, ["awa", {"qwq": "qwq"}])
