"""Terminal styling helpers + arrow-key selection picker.

Stdlib only. Falls back to numbered input on non-TTY, Windows, or NO_COLOR.
"""
import os
import re
import sys


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_INTERACTIVE = sys.stdin.isatty() and sys.stdout.isatty()

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


_BRAND_TEAL = "38;2;77;182;182"   # ~#4DB6B6
_BRAND_BLUE = "38;2;92;122;184"   # ~#5C7AB8


def brand(text: str) -> str: return _style(_BRAND_TEAL, text)
def brand_alt(text: str) -> str: return _style(_BRAND_BLUE, text)


def _monocle_wordmark() -> str:
    if not _USE_COLOR:
        return "Monocle"
    return "\033[1;{}mMon\033[0m\033[1;{}mocle\033[0m".format(_BRAND_TEAL, _BRAND_BLUE)


def header(title: str, subtitle: str = "") -> None:
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


def select(question: str, options: list, links: list = None) -> str:
    """Arrow-key single-select. Each option: {key, label, hint?}.
    links: [(label, url), ...] shown below the menu. Returns the chosen key."""
    if not _INTERACTIVE or not _ARROW_KEYS_AVAILABLE:
        return _select_numeric(question, options, links)
    return _select_arrow(question, options, links)


def _link_label_width(links: list) -> int:
    return max(len(label) for label, _ in links) if links else 0


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _strip_ansi(s: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', s)


def _physical_row_count(logical_lines: list, term_width: int) -> int:
    """Physical rows occupied by logical_lines at term_width.
    Needed after resize — the terminal reflows content so logical != physical."""
    if term_width <= 0:
        term_width = 1
    count = 0
    for line in logical_lines:
        visible = len(_strip_ansi(line))
        count += max(1, (visible + term_width - 1) // term_width)
    return count


def _select_numeric(question: str, options: list, links: list = None) -> str:
    section(question)
    for line in _option_lines(options, -1):
        print(line)
    print()
    if links:
        width = _link_label_width(links)
        for label, url in links:
            print("  " + dim(label.ljust(width) + "  ·  ") + brand_alt(url))
        print()
    valid = [str(i) for i in range(1, len(options) + 1)]
    while True:
        choice = input("  Choose " + dim("[" + "/".join(valid) + "]") + " " + brand("›") + " ").strip()
        if choice in valid:
            return options[int(choice) - 1]["key"]


def _option_lines(options: list, selected: int) -> list:
    # Pad labels to the widest so hints align. Truncate hints to terminal width
    # so lines never wrap — wrapping breaks cursor math during redraws.
    label_width = max(len(opt["label"]) for opt in options)
    hint_budget = _terminal_width() - (8 + label_width) - 5
    lines = []
    for i, opt in enumerate(options):
        label = opt["label"]
        if i == selected:
            marker, num, styled_label = brand("›"), brand(str(i + 1)), bold(label)
        else:
            marker, num, styled_label = " ", dim(str(i + 1)), label
        pad = " " * (label_width - len(label))
        line = "  " + marker + " " + num + "   " + styled_label + pad
        if opt.get("hint") and hint_budget >= 8:
            hint = opt["hint"]
            if len(hint) > hint_budget:
                hint = hint[:hint_budget - 1] + "…"
            line += "  " + dim("—  " + hint)
        lines.append(line)
    return lines


def _select_arrow(question: str, options: list, links: list = None) -> str:
    selected = 0
    section(question)

    _nav = "  " + dim("↑↓ navigate  ·  enter select  ·  ctrl-c cancel")
    _link_lines = []
    if links:
        width = _link_label_width(links)
        for label, url in links:
            _link_lines.append("  " + dim(label.ljust(width) + "  ·  ") + brand_alt(url))
    _footer = [""] + [_nav] + _link_lines

    lines = _option_lines(options, selected)
    for line in lines:
        print(line)
    for footer_line in _footer:
        print(footer_line)

    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old_settings = _termios.tcgetattr(fd)
    try:
        # setcbreak keeps OPOST so \n → \r\n; raw mode would misalign redraws.
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

                # Recompute physical rows at current width — a resize reflows
                # old content, so logical line count is no longer enough.
                term_w = _terminal_width()
                phys_up = (_physical_row_count(lines, term_w)
                           + _physical_row_count(_footer, term_w))
                sys.stdout.write("\x1b[{}A\r\x1b[J".format(phys_up))
                lines = _option_lines(options, selected)
                for line in lines:
                    sys.stdout.write(line + "\n")
                for footer_line in _footer:
                    sys.stdout.write(footer_line + "\n")
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

    # Collapse section + menu + footer into a single confirmation line.
    term_w = _terminal_width()
    phys_total = (_physical_row_count(["  " + bold(question), ""], term_w)
                  + _physical_row_count(lines, term_w)
                  + _physical_row_count(_footer, term_w))
    sys.stdout.write("\x1b[{}A\r\x1b[J".format(phys_total))
    chosen = options[selected]
    print("  " + brand("✓") + " " + chosen["label"] + dim("  ·  " + question))
    sys.stdout.flush()
    return chosen["key"]
