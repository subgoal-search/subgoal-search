import collections
import re


class LoggerWithSmoothing:
    def __init__(self):
        self._previous_values = collections.defaultdict(float)
        self._smoothing_coeffs = collections.defaultdict(float)

    def _log_scalar(self, name, step, value):
        raise NotImplementedError()

    def log_scalar(self, name, step, value, smoothing=None):
        self._log_scalar(name, step, value)
        if smoothing is not None:
            smoothing_regex, smoothing_coeff = smoothing
            if re.search(smoothing_regex, name) is not None:
                name_smoothed = name + rf'/smoothing_{smoothing_coeff}'
                prev_value = self._previous_values[name_smoothed]
                prev_smoothing_coeff = self._smoothing_coeffs[name_smoothed]

                new_smoothing_coeff = prev_smoothing_coeff * 0.9 + smoothing_coeff * 0.1
                new_value = value * (1 - prev_smoothing_coeff) + prev_value * prev_smoothing_coeff

                self._previous_values[name_smoothed] = new_value
                self._smoothing_coeffs[name_smoothed] = new_smoothing_coeff
                self._log_scalar(name_smoothed, step, new_value)


class StdoutLogger(LoggerWithSmoothing):
    def _log_scalar(self, name, step, value):
        print('{:>6} | {:32}{:>9.3f}'.format(step, name + ':', value))

    @staticmethod
    def log_property(name, value):
        pass


_loggers = [StdoutLogger()]


def register_logger(logger):
    _loggers.append(logger)


def log_scalar(name, step, value, smoothing=None):
    for logger in _loggers:
        logger.log_scalar(name, step, value, smoothing)


def log_property(name, value):
    for logger in _loggers:
        logger.log_property(name, value)


def log_scalar_metrics(prefix, step, metrics, smoothing=None):
    for (name, value) in metrics.items():
        log_scalar(prefix + '/' + name, step, value, smoothing=smoothing)
