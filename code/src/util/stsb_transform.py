import numpy as np

def stsb_score_to_class_index(score):
    """
    given some stsb score (e.g. from the dataset), output a class index e ranging from 0 to 21
    each index represents an increment, 0 maps to a score of 1, 1 maps to a score of 1.2 and so on
    """
    rounded = round(float(score)*5)/5
    # clip the rounded value for good measure
    rounded = np.clip(rounded, 1, 5)
    # return its class index
    return round((rounded - 1.0) / 0.2)

def stsb_class_index_to_score(idx):
    return (1 + (idx*0.2))


if __name__ == "__main__":
    # do a little testing
    assert(stsb_score_to_class_index("0") == 0)
    assert(stsb_score_to_class_index(0) == 0)

    assert(stsb_score_to_class_index(1.2) == 1)
    assert(stsb_score_to_class_index(4.8) == 19)
    assert(stsb_score_to_class_index(5.0) == 20)

    # clipping sould also work
    assert(stsb_score_to_class_index(0) == 0)
    assert(stsb_score_to_class_index(5.2) == 20)

    assert(stsb_class_index_to_score(0) == 1.0)
    assert(stsb_class_index_to_score(20) == 5.0)