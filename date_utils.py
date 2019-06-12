import datetime


def reddit_utc_string_to_unix_timestamp(string):
    return int(datetime.datetime.strptime(string, "%a %b %d %H:%M:%S %Y UTC").timestamp())
