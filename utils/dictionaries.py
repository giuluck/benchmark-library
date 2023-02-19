def merge(*dictionaries: dict) -> dict:
    """Merges multiple dictionaries. In case of key clash, keeps the value of the earliest dictionary.

    :param dictionaries:
        A list of dictionaries to merge.

    :return:
        The merged dictionary.
    """
    output = {}
    for d in dictionaries[::-1]:
        output.update(d)
    return output
