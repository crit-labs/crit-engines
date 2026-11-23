"""Gemma 4's tool-call syntax (tool_format.py): blocks as the chat template writes them parse back into the call's
name and JSON arguments, crit_core.text.tools' parser splits a reply in that format into content and calls, and a
block that does not parse whole, names an unoffered function or repeats a key stays no call. CPU only."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gemma4_tool_format", HERE / "tool_format.py")
tool_format = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_format)

Q = '<|"|>'
FUNCTIONS = {"read_file": {"name": "read_file"}, "run": {"name": "run"}}


def block(body):
    return tool_format.OPENING + body + tool_format.CLOSING


class ParseTests(unittest.TestCase):
    def test_blocks_as_the_template_writes_them(self):
        cases = [
            (f"call:read_file{{path:{Q}docs/README.md{Q}}}", ("read_file", {"path": "docs/README.md"})),
            ("call:run{}", ("run", {})),
            (f"call:run{{args:[{Q}-q{Q},{Q}tests{Q}],env:{{CUDA:{Q}0{Q},depth:2}},retries:3,ratio:0.25,"
             f"strict:true,timeout:null,verbose:false}}",
             ("run", {"args": ["-q", "tests"], "env": {"CUDA": "0", "depth": 2}, "retries": 3, "ratio": 0.25,
                      "strict": True, "timeout": None, "verbose": False})),
            # a string is written verbatim: braces, commas, colons and quotes inside it are text
            (f'call:run{{command:{Q}echo "a, b: {{c}}" [x]{Q}}}', ("run", {"command": 'echo "a, b: {c}" [x]'})),
            (f"call:run{{n:-1.5e-3,m:[]}}", ("run", {"n": -0.0015, "m": []})),
            (f"call:run{{{Q}quoted key{Q}:1}}", ("run", {"quoted key": 1})),
            (f"  call:read_file{{path:{Q}a{Q}}}\n", ("read_file", {"path": "a"})),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(tool_format.parse(body, FUNCTIONS), expected)

    def test_what_is_no_call(self):
        for body in [
            f"call:write_file{{path:{Q}a{Q}}}",        # not offered
            f"call:run{{path:{Q}a{Q},path:{Q}b{Q}}}",  # a repeated key
            f"call:run{{path:{Q}a}}",                  # an unclosed string
            "call:run{path:a}",                        # a bare word is no value
            "call:run{n:1",                            # unclosed arguments
            "call:run{n:1}trailing",                   # more after the arguments
            "run{n:1}",                                # no call: prefix
            "call:run{n:01}",                          # no number as JSON writes one
            "call:run{n:truex}",
            "",
        ]:
            with self.subTest(body=body):
                self.assertIsNone(tool_format.parse(body, FUNCTIONS))

    def test_the_format_names_the_template_s_markers(self):
        format = tool_format.FORMAT
        self.assertEqual((format.opening, format.closing, format.markers), ("<|tool_call>", "<tool_call|>", (Q,)))


class StreamTests(unittest.TestCase):
    BODIES = [f"call:run{{args:[{Q}-q{Q},{Q}tests{Q}],env:{{CUDA:{Q}0{Q},depth:2}},retries:13,ratio:0.25,"
              f"strict:true,timeout:null}}",
              f'call:read_file{{path:{Q}a, b: {{c}}{Q}}}',
              "call:run{}"]

    def test_every_prefix_streams_a_prefix_of_the_whole_call(self):
        import json
        for body in self.BODIES:
            name, arguments = tool_format.parse(body, FUNCTIONS)
            whole = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            seen = ""
            for end in range(len(body) + 1):
                found = tool_format.stream(body[:end], FUNCTIONS)
                if found is None:
                    self.assertFalse(seen, (body, end))
                    continue
                self.assertEqual(found[0], name)
                self.assertTrue(whole.startswith(found[1]), (body[:end], found[1], whole))
                self.assertTrue(found[1].startswith(seen), (body[:end], found[1], seen))
                seen = found[1]
            self.assertEqual(seen, whole[:-1] if arguments else "{")

    def test_a_reply_cut_inside_a_call_keeps_the_call(self):
        """As crit serve answers a reply its output limit cut inside the call: the call with the arguments closed so
        far, not the block as text (Eval compares such a reply with the whole one)."""
        import json
        from crit_core.text.tools import ToolReply
        request = {"tools": [{"type": "function", "function": spec} for spec in FUNCTIONS.values()]}
        reply = ToolReply(tool_format.FORMAT, request)
        text = tool_format.OPENING + f"call:run{{args:[{Q}ls{Q}],retries:3,command:{Q}echo hel"
        for start in range(0, len(text), 6):
            reply.push(text[start:start + 6])
        content, calls = reply.finish(cut=True)
        self.assertEqual(content, "")
        self.assertEqual([(c["function"]["name"], c["function"]["arguments"]) for c in calls],
                         [("run", '{"args": ["ls"], "retries": 3')])
        whole = ToolReply(tool_format.FORMAT, request)
        whole.push(text + f"lo{Q}}}" + tool_format.CLOSING)
        _, calls = whole.finish()
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"args": ["ls"], "retries": 3, "command": "echo hello"})


class ReasoningTests(unittest.TestCase):
    def test_the_thinking_channel_is_reasoning_not_reply(self):
        from crit_core.text.reasoning import ReasoningSplitter
        splitter = ReasoningSplitter(tool_format.THINKING)
        events = []
        reply = "<|channel>thought\n<channel|>The tide rises twice a day."
        for start in range(0, len(reply), 5):
            events += splitter.push(reply[start:start + 5])
        events += splitter.finish()
        text = "".join(v for kind, v in events if kind == "text")
        self.assertEqual(text, "The tide rises twice a day.")


class ReplyTests(unittest.TestCase):
    def test_crit_splits_a_reply_into_content_and_calls(self):
        from crit_core.text.tools import ToolCallParser
        tools = [{"type": "function", "function": spec} for spec in FUNCTIONS.values()]
        reply = ("I will read it." + block(f"call:read_file{{path:{Q}README.md{Q}}}")
                 + block(f"call:run{{args:[{Q}ls{Q}]}}"))
        parser = ToolCallParser(tool_format.FORMAT, tools)
        events = []
        for start in range(0, len(reply), 7):   # streamed in pieces that cut markers apart
            events += parser.push(reply[start:start + 7])
        events += parser.finish()
        text = "".join(value for kind, value in events if kind == "text")
        calls = [value for kind, value in events if kind == "call"]
        self.assertEqual(text, "I will read it.")
        self.assertEqual([(call["name"], call["arguments"]) for call in calls],
                         [("read_file", {"path": "README.md"}), ("run", {"args": ["ls"]})])


if __name__ == "__main__":
    unittest.main()
