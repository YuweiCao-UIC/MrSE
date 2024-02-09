
def decode(division):
    if type(division) is dict:
        prediction_dict = {m: event for event, messages in division.items() for m in messages}
    elif type(division) is list:
        prediction_dict = {m: event for event, messages in enumerate(division) for m in messages}
    prediction_dict_sorted = dict(sorted(prediction_dict.items()))
    return list(prediction_dict_sorted.values())