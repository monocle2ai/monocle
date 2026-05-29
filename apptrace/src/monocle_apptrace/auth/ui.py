"""Terminal styling helpers + arrow-key selection picker.

Stdlib only. ANSI escapes for color, termios/tty for raw input. Falls back
to numbered input when stdin isn't a TTY, on Windows, or when `NO_COLOR`
is set. Glyphs: `>` header, `›` cursor, `✓` confirm, `•` step.
"""
import os
import sys


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_INTERACTIVE = sys.stdin.isatty() and sys.stdout.isatty()

# Probe once for the raw-keystroke modules used by the arrow-key picker.
# These ship with every CPython on POSIX but not on Windows.
try:
    import termios as _termios
    import tty as _tty
    _ARROW_KEYS_AVAILABLE = True
except ImportError:
    _ARROW_KEYS_AVAILABLE = False


def _style(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return "\033[{}m{}\033[0m".format(code, text)


def bold(text: str) -> str: return _style("1", text)
def dim(text: str) -> str: return _style("2", text)
def red(text: str) -> str: return _style("31", text)


# Monocle brand palette — teal "Mon" + blue "ocle" from the logo.
# 24-bit truecolor; degrades to plain text on NO_COLOR or non-TTY.
_BRAND_TEAL = "38;2;77;182;182"   # ~#4DB6B6
_BRAND_BLUE = "38;2;92;122;184"   # ~#5C7AB8


def brand(text: str) -> str: return _style(_BRAND_TEAL, text)
def brand_alt(text: str) -> str: return _style(_BRAND_BLUE, text)


def _monocle_wordmark() -> str:
    if not _USE_COLOR:
        return "Monocle"
    return "\033[1;{}mMon\033[0m\033[1;{}mocle\033[0m".format(_BRAND_TEAL, _BRAND_BLUE)


def header(title: str, subtitle: str = "") -> None:
    # Treat the literal word "Monocle" specially so the wordmark gets the
    # brand gradient while the rest of the title stays default bold.
    if title.startswith("Monocle "):
        styled_title = _monocle_wordmark() + bold(title[len("Monocle"):])
    else:
        styled_title = bold(title)
    line = "> " + styled_title
    if subtitle:
        line += "  " + dim("·") + "  " + dim(subtitle)
    print()
    print(line)
    print()


def section(text: str) -> None:
    print("  " + bold(text))
    print()


def step(text: str) -> None:
    print("  " + dim("•") + " " + text)


def check(text: str) -> None:
    print("  " + brand("✓") + " " + text)


def fail(text: str) -> None:
    # Red is intentional for errors — universal "broke" indicator.
    print("  " + red("✗") + " " + text)


def hint(text: str) -> None:
    print("    " + dim(text))


def blank() -> None:
    print()


def confirm(question: str, default_yes: bool = True) -> bool:
    if not _INTERACTIVE:
        return default_yes
    suffix = "[Y/n]" if default_yes else "[y/N]"
    response = input("  " + question + " " + dim(suffix) + " " + brand("›") + " ").strip().lower()
    if not response:
        return default_yes
    return response.startswith("y")


# ---------------------------------------------------------------------------
# Interactive selection — arrow keys with numeric fallback.
# ---------------------------------------------------------------------------


def select(question: str, options: list, docs_url: str = None) -> str:
    """Single-select picker. Each option is a dict with `key`, `label`, and
    optional `hint`. Returns the chosen `key`. Falls back to numeric input
    when not running on an interactive POSIX TTY."""
    if not _INTERACTIVE or not _ARROW_KEYS_AVAILABLE:
        return _select_numeric(question, options, docs_url)
    return _select_arrow(question, options, docs_url)


def _select_numeric(question: str, options: list, docs_url: str = None) -> str:
    section(question)
    for line in _option_lines(options, -1):  # -1 → nothing highlighted in fallback
        print(line)
    print()
    if docs_url:
        print("  " + dim("Learn more  ·  ") + brand_alt(docs_url))
        print()
    valid = [str(i) for i in range(1, len(options) + 1)]
    while True:
        choice = input("  Choose " + dim("[" + "/".join(valid) + "]") + " " + brand("›") + " ").strip()
        if choice in valid:
            return options[int(choice) - 1]["key"]


def _option_lines(options: list, selected: int) -> list:
    # Labels padded to the widest one so the em-dash before each hint aligns.
    # `selected = -1` highlights nothing (used by the numeric fallback).
    label_width = max(len(opt["label"]) for opt in options)
    lines = []
    for i, opt in enumerate(options):
        label = opt["label"]
        if i == selected:
            marker = brand("›")
            num = brand(str(i + 1))
            styled_label = bold(label)
        else:
            marker = " "
            num = dim(str(i + 1))
            styled_label = label
        pad = " " * (label_width - len(label))
        line = "  " + marker + " " + num + "   " + styled_label + pad
        if opt.get("hint"):
            line += "  " + dim("—  " + opt["hint"])
        lines.append(line)
    return lines


def _select_arrow(question: str, options: list, docs_url: str = None) -> str:
    selected = 0
    section(question)
    lines = _option_lines(options, selected)
    for line in lines:
        print(line)
    print()
    print("  " + dim("↑↓ navigate  ·  enter select  ·  ctrl-c cancel"))
    if docs_url:
        print("  " + dim("learn more  ·  ") + brand_alt(docs_url))

    menu_height = len(lines)
    footer_height = 3 if docs_url else 2  # blank + nav (+ docs line)
    # Hide cursor while the menu is interactive (avoids blink at footer end).
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old_settings = _termios.tcgetattr(fd)
    try:
        # `setcbreak` (not setraw): keeps OPOST so \n still translates to
        # \r\n. Raw mode would cause each redraw line to start mid-row.
        _tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    selected = (selected - 1) % len(options)
                elif seq == "[B":
                    selected = (selected + 1) % len(options)
                else:
                    continue
                # Cursor is just past the footer. Walk up over (footer + menu),
                # rewrite menu lines in place, walk back down to the footer.
                sys.stdout.write("\x1b[{}A\r".format(footer_height + menu_height))
                for line in _option_lines(options, selected):
                    sys.stdout.write("\x1b[2K" + line + "\n")
                sys.stdout.write("\x1b[{}B\r".format(footer_height))
                sys.stdout.flush()
            elif ch in ("\r", "\n"):
                break
            elif ch == "\x03":
                _termios.tcsetattr(fd, _termios.TCSADRAIN, old_settings)
                sys.stdout.write("\x1b[?25h")
                sys.stdout.flush()
                print()
                raise KeyboardInterrupt
    finally:
        _termios.tcsetattr(fd, _termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

    # Collapse menu+footer into one confirmation line so the screen reads
    # as a clean trail of resolved choices.
    #   section header + blank = 2 lines  (always)
    #   menu                   = menu_height
    #   blank + nav (+ docs)   = footer_height
    section_height = 2
    total_lines = section_height + menu_height + footer_height
    sys.stdout.write("\x1b[{}A\r".format(total_lines))
    for _ in range(total_lines):
        sys.stdout.write("\x1b[2K\n")
    sys.stdout.write("\x1b[{}A\r".format(total_lines))
    chosen = options[selected]
    print("  " + brand("✓") + " " + chosen["label"] + dim("  ·  " + question))
    sys.stdout.flush()
    return chosen["key"]
