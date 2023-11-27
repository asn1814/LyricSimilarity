"""
Module Docstring
"""

import pandas as pd

__author__ = "Andrew Nakamoto"


def main():
    """ Main entry point of the app """
    df = pd.read_csv('topSongsLyrics1950_2019.csv')
    print(df)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()