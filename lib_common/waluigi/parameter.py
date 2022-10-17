# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
import luigi


class MyListParameter(luigi.ListParameter):
    @classmethod
    def parse(cls, x):
        # for avoiding many escapes \ for string list on shell and adding []
        # assumes x is list of str without " and [] or list of number
        if isinstance(x, list):
            return x
        elif all([not xx.isdigit() for xx in x]):
            x = '["' + x.replace(',', '","') + '"]'
        else:
            x = '[' + x + ']'
        return super().parse(cls, x)
