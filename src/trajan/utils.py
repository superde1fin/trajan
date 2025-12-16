import argparse
import sys

from . import constants

_ = lambda s: s


class StrictSubParsersAction(argparse._SubParsersAction):

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        arg_strings = values[1:]

        # set the parser name if requested
        if self.dest is not argparse.SUPPRESS:
            setattr(namespace, self.dest, parser_name)

        # select the parser
        try:
            subparser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
            raise ArgumentError(self, msg)


        #Parse argse with error calls from subparser instead of delegating
        #unrecognized argument error to the main parser
        subnamespace = subparser.parse_args(arg_strings)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)



class NoMetavarHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _get_default_metavar_for_optional(self, action):
        return ""

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        return "%s" % get_metavar(1)


class ErrorHandlingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def error(self, message):
        sys.stderr.write("\n\nERROR: %s\n\n" % message)
        self.print_help()
        sys.exit(2)

def verbosity_type():
    def checker(value):
        ivalue = int(value)
        if ivalue < 1 or ivalue > constants.MAX_VERBOSITY:
            raise argparse.ArgumentTypeError(f"Verbosity must be between 1 and {constants.MAX_VERBOSITY}")
        return ivalue
    return checker
