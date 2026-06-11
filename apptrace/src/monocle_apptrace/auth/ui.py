"""Terminal styling helpers + arrow-key selection picker.

Stdlib only. ANSI escapes for color, termios/tty for raw input. Falls back
to numbered input when stdin isn't a TTY, on Windows, or when `NO_COLOR`
is set. Glyphs: `>` header, `›` cursor, `✓` confirm, `•` step.
"""
import os
import re
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


def select(question: str, options: list, links: list = None) -> str:
    """Single-select picker. Each option is a dict with `key`, `label`, and
    optional `hint`. `links` is an optional list of (label, url) tuples shown
    under the menu. Returns the chosen `key`. Falls back to numeric input
    when not running on an interactive POSIX TTY."""
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
    """Physical terminal rows occupied by a list of logical lines at the given width.

    Used to correct cursor-movement offsets after terminal resize: when the
    window gets narrower the terminal reflows previously-printed content so
    each logical line may occupy more than one physical row.
    """
    if term_width <= 0:
        term_width = 1
    count = 0
    for line in logical_lines:
        visible = len(_strip_ansi(line))
        count += max(1, (visible + term_width - 1) // term_width)
    return count


def _select_numeric(question: str, options: list, links: list = None) -> str:
    section(question)
    for line in _option_lines(options, -1):  # -1 → nothing highlighted in fallback
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
    # Labels padded to the widest one so the em-dash before each hint aligns.
    # `selected = -1` highlights nothing (used by the numeric fallback).
    label_width = max(len(opt["label"]) for opt in options)
    # Visible prefix width before the hint text:
    #   "  " + marker(1) + " " + num(1) + "   " + label_width = 8 + label_width
    # Hint separator "  —  " = 5 visible chars.
    # Truncate hint so the full line never wraps — wrapping breaks the cursor
    # arithmetic used during arrow-key redraws and causes doubled output.
    hint_budget = _terminal_width() - (8 + label_width) - 5
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

    # Build footer strings once so we can measure their physical rows later.
    _nav = "  " + dim("↑↓ navigate  ·  enter select  ·  ctrl-c cancel")
    _link_lines = []
    if links:
        width = _link_label_width(links)
        for label, url in links:
            _link_lines.append("  " + dim(label.ljust(width) + "  ·  ") + brand_alt(url))
    # footer = blank line + nav hint + optional link lines
    _footer = [""] + [_nav] + _link_lines

    lines = _option_lines(options, selected)
    for line in lines:
        print(line)
    for footer_line in _footer:
        print(footer_line)

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

                # Re-measure at the *current* terminal width so that a resize
                # between the initial render and this keypress is handled
                # correctly. The terminal reflows old content to the new width,
                # so physical rows = f(visible_chars, current_width).
                term_w = _terminal_width()
                phys_up = (_physical_row_count(lines, term_w)
                           + _physical_row_count(_footer, term_w))

                # Move to the start of the menu, clear everything below
                # (handles stale wrapped rows left by the old render), then
                # reprint the menu and footer at the new selection.
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

    # Collapse the entire menu block (section header + menu + footer) into one
    # confirmation line so the screen reads as a clean trail of resolved choices.
    term_w = _terminal_width()
    section_lines = ["  " + bold(question), ""]  # matches what section() prints
    phys_total = (_physical_row_count(section_lines, term_w)
                  + _physical_row_count(lines, term_w)
                  + _physical_row_count(_footer, term_w))
    sys.stdout.write("\x1b[{}A\r\x1b[J".format(phys_total))
    chosen = options[selected]
    print("  " + brand("✓") + " " + chosen["label"] + dim("  ·  " + question))
    sys.stdout.flush()
    return chosen["key"]
