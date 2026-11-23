"""Gemma 4's tool calls, as its chat template writes them (standard library and crit_core only).

The template renders an assistant's call as <|tool_call>call:NAME{KEY:VALUE,...}<tool_call|>: the arguments in
key order, a key bare, a value by format_argument with escape_keys=False: a string between <|"|> markers (written
verbatim, nothing escaped), null, true, false, a number as Python writes it, a list [VALUE,...] and an object
{KEY:VALUE,...} with bare keys. The model writes its calls the same way, so this parses that grammar back into the
call's name and JSON arguments. A block that does not parse whole, names a function the request does not offer or
repeats a key is no call (None), and crit_core.text.tools leaves it in the reply as written. <|tool_call>,
<tool_call|> and <|"|> are special tokens of the tokenizer, kept in the reply text so the block can be parsed.

THINKING is Gemma 4's reasoning: the model writes its thinking as <|channel>thought
...<channel|> (an empty block when
thinking is off), two special tokens crit_core.text.reasoning does not register. Without it the markers are dropped
from the reply text and the channel's name, "thought", reaches the reply."""
import json
import re

from crit_core.text.reasoning import ReasoningFormat
from crit_core.text.tools import ToolFormat

OPENING, CLOSING, QUOTE = "<|tool_call>", "<tool_call|>", '<|"|>'
_NAME = re.compile(r"call:([A-Za-z0-9_.\-]{1,64})\{")
_KEY = re.compile(r"[^\s:,{}\[\]<>]+")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_WORDS = {"true": True, "false": False, "null": None}


class _Invalid(ValueError):
    pass


class _Reader:
    def __init__(self, text, at):
        self.text, self.at = text, at

    def peek(self, literal):
        return self.text.startswith(literal, self.at)

    def take(self, literal):
        if not self.peek(literal):
            raise _Invalid(literal)
        self.at += len(literal)

    def string(self):
        self.take(QUOTE)
        end = self.text.find(QUOTE, self.at)
        if end < 0:
            raise _Invalid("unclosed string")
        value, self.at = self.text[self.at:end], end + len(QUOTE)
        return value

    def key(self):
        if self.peek(QUOTE):   # format_argument's escaped keys, should a model write them
            return self.string()
        match = _KEY.match(self.text, self.at)
        if not match:
            raise _Invalid("key")
        self.at = match.end()
        return match.group()

    def members(self, closing):
        """KEY:VALUE pairs up to `closing`, which it consumes."""
        found = {}
        if self.peek(closing):
            self.at += len(closing)
            return found
        while True:
            key = self.key()
            if key in found:
                raise _Invalid("repeated key")
            self.take(":")
            found[key] = self.value()
            if self.peek(","):
                self.at += 1
                continue
            self.take(closing)
            return found

    def value(self):
        if self.peek(QUOTE):
            return self.string()
        if self.peek("{"):
            self.at += 1
            return self.members("}")
        if self.peek("["):
            self.at += 1
            items = []
            if self.peek("]"):
                self.at += 1
                return items
            while True:
                items.append(self.value())
                if self.peek(","):
                    self.at += 1
                    continue
                self.take("]")
                return items
        for word, value in _WORDS.items():
            if self.peek(word):
                self.at += len(word)
                return value
        match = _NUMBER.match(self.text, self.at)
        if not match:
            raise _Invalid("value")
        self.at = match.end()
        number = match.group()
        return int(number) if re.fullmatch(r"-?[0-9]+", number) else float(number)


def parse(body, functions):
    """(name, arguments) for one block body `call:NAME{...}` whose NAME is one of `functions`, else None."""
    text = body.strip()
    match = _NAME.match(text)
    if not match or match.group(1) not in functions:
        return None
    reader = _Reader(text, match.end())
    try:
        arguments = reader.members("}")
    except _Invalid:
        return None
    return (match.group(1), arguments) if reader.at == len(text) else None


def stream(body, functions):
    """(name, the start of its arguments as JSON) for an unfinished block body, or None until the name is written: the
    members whose value a delimiter after it has closed, serialized as json.dumps(arguments, ensure_ascii=False) writes
    them, so the text begins every later return and the whole block's arguments ("{" while none is closed). A call cut
    by its output limit then keeps them, as OpenAI's API returns a call cut by finish_reason "length"."""
    text = body.lstrip()
    match = _NAME.match(text)
    if not match or match.group(1) not in functions:
        return None
    reader = _Reader(text, match.end())
    done = []
    try:
        if reader.peek("}"):
            return match.group(1), "{"
        while True:
            key = reader.key()
            reader.take(":")
            value = reader.value()
            if reader.at >= len(text):
                break                           # the value may go on (a number, a word) until a delimiter follows
            if reader.peek(",") or reader.peek("}"):
                if key in (k for k, _ in done):
                    return None
                done.append((key, value))
            if not reader.peek(","):
                break
            reader.at += 1
    except _Invalid:
        pass                                    # the text ends inside a member: the members before it stand
    return match.group(1), "{" + ", ".join(json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
                                           for k, v in done)


FORMAT = ToolFormat(parse, OPENING, CLOSING, markers=(QUOTE,), stream=stream)
THINKING = ReasoningFormat("<|channel>", "<channel|>")
