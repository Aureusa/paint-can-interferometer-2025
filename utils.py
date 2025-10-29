def print_box(message: str):
    """
    Print a message in a box format.
    The box is created using Unicode box-drawing characters.

    :param message: The message to be printed in the box.
    :type message: str
    """
    # Ensure the box starts on a new line
    print()

    #91
    lines = message.split('\n')
    for line in lines:
        if len(line) > 91:
            split_index = line.rfind(' ', 0, 91)
            if split_index != -1:
                lines.insert(lines.index(line) + 1, line[split_index + 1:])
                lines[lines.index(line)] = line[:split_index]
    max_length = 88
    border_up = '┌' + '─' * (max_length + 2) + '┐'
    border_down = '└' + '─' * (max_length + 2) + '┘'
    print(border_up)
    for line in lines:
        print(f'│ {line.ljust(max_length)} │')
    print(border_down)
