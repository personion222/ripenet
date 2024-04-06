def color_text(txt, bg=None, fg=None):
    if bool(bg) != bool(fg):
        if fg:
            color_txt = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m{txt}\033[0m"

        else:
            color_txt = f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{txt}\033[0m"

    elif bg is None:
        color_txt = txt

    else:
        color_txt = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{txt}\033[0m"

    return color_txt


print(color_text("Hello World!", bg=(15, 70, 127), fg=(30, 140, 255)))
