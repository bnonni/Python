"""
SpotTheDifference.py features a single function called spot_the_difference that takes 2 arguments.
The arguments are 2 lists of strings. The function interleaves the arrays, adjusting for length,
and then compares elements next to one another, adding small pair arrays to another array to form
a 2D array. White spaces are preserved. The goal is the easily spot differences between the
two lists once the 2D array is printed.
"""
from more_itertools import roundrobin

def spot_the_difference(conversation_1: list, conversation_2: list) -> list(list()):
    """
    Filters adds each word from each list into pairs
    (preserving order) and prints each one to the terminal

    Parameters
    ----------
    conversation_1 : list(str)
        the first conversation you wish to compare to conversation_2

    conversation_2 : list(str)
        the second conversation you wish to compare to conversation_1
        
    Returns
    -------
    comparison_matrix : list(list())
        2D array containing all matches between each conversation and
        all unique words from each conversation
    """

    # Set a boolean value to True/False if conversation 2 is/is not longer
    conversation_2_longer = len(conversation_1) < len(conversation_2)

    # Set a boolean value to True/False if converstaions are equal/not equal
    conversation_length_equality = len(conversation_1) == len(conversation_2)

    # Check for inequality in converstaion length
    if not conversation_length_equality:
        # Determine which conversation is shorter, use difference in loop
        # to pad shorter conversation with empty spaces
        if conversation_2_longer:
            length_difference = len(conversation_2) - len(conversation_1)
            for i in range(length_difference):
                conversation_1.append(" ")
        else:
            length_difference = len(conversation_1) - len(conversation_2)
            for i in range(length_difference):
                conversation_2.append(" ")

    # Use more_itertools method roundrobin to interleave the
    # the two conversations, starting with the longer one
    if conversation_length_equality:
        master_list = list(roundrobin(conversation_1, conversation_2))
    elif conversation_2_longer:
        master_list = list(roundrobin(conversation_2, conversation_1))
    else:
        master_list = list(roundrobin(conversation_1, conversation_2))

    # Set a variable to a 1D array to add arrays to later
    comparison_matrix = []
    # Set a pointer to make indexing easier (i.e. looking ahead)
    i = 0
    while i < len(master_list) - 1:
        if master_list[i] == master_list[i+1]:
            comparison_matrix.append([master_list[i], master_list[i+1]])
            i+=2
        else:
            comparison_matrix.append([master_list[i], ""])
            i+=1

    return comparison_matrix

if __name__ == '__main__':
    c1 = ["Hi","Howdy","How are you?","...Hope you are well, haven't heard from you in a while.",
          "Doing fine, you?","Not so bad. What's up?","Nothing much.",
          "Well, it can be boring during quarantine.","Yup. What did you do yesterday?",
          "Nothing much.", "Oh well. Be seeing ya.","Cya"]

    c2 = ["John?","Hi","Howdy","How are you?","Doing fine, you?",
          "Not so bad. What's up?","I've been busy coding the whole weekend.",
          "Well, it can be boring during quarantine.","Yeah... What did you do yesterday?",
          "Nothing much.","Oh well. Be seeing ya.","Cya","Cya."]

    comparison = spot_the_difference(c1, c2)
    for pair in comparison:
        print(pair)
