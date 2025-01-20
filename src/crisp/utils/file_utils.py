import logging
import os
import pathlib
import random
import string


def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
    """Credit: https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits"""
    return "".join(random.choice(chars) for _ in range(size))


def uniquify(path):
    """If path exists, append suffix
    Credit: https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def safely_make_folders(folders):
    if type(folders) is str:
        folders = [folders]
    for fpath in folders:
        pathlib.Path(fpath).mkdir(parents=True, exist_ok=True)


def assert_folders_exist(folders):
    for folder in folders:
        if not os.path.exists(folder):
            raise ValueError(f"{folder} does not exist.")
        if not os.path.isdir(folder):
            raise ValueError(f"{folder} is not a folder.")


def assert_files_exist(fpaths):
    for path in fpaths:
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")


def set_up_logger(log_stream_list, log_file_list, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)

    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s")

    handlers = []
    for log_file in log_file_list:
        flog_handler = logging.FileHandler(log_file)
        flog_handler.setFormatter(log_formatter)
        handlers.append(flog_handler)

    for log_stream in log_stream_list:
        tlog_handler = logging.StreamHandler(log_stream)
        tlog_handler.setFormatter(log_formatter)
        handlers.append(tlog_handler)

    logger.handlers = handlers
