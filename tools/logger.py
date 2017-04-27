import sys


class Logger(object):
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
