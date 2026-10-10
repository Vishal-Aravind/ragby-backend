"""Website flow engine tests. No network, no database.

Run from backend/:  python -m unittest tests.test_web_flows -v
"""
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone

from web_flows.templating import render_text, render_url, set_variable, to_decimal
from web_flows.engine import Engine, MAX_AUTO_STEPS
from flow_common import option_id

PID = "11111111-1111-1111-1111-111111111111"
VID = "visitor-1"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeStore:
    def __init__(self, flow, nodes, edges):
        self.flow = flow
        self.nodes = {n["id"]: n for n in nodes}
        self.edges = edges
        self.sessions = {}

    def get_active_web_flow(self, project_id):
        return copy.deepcopy(self.flow) if self.flow and project_id == PID else None

    def get_start_node(self, flow_id):
        return next((copy.deepcopy(n) for n in self.nodes.values() if n.get("is_start")), None)

    def get_node(self, node_id, flow_id):
        n = self.nodes.get(node_id)
        return copy.deepcopy(n) if n else None

    def get_edges(self, flow_id):
        return copy.deepcopy(self.edges)

    def load_session(self, chat_id):
        s = self.sessions.get(chat_id)
        return copy.deepcopy(s) if s else None

    def create_session(self, sess):
        self.sessions[sess["chat_id"]] = copy.deepcopy(sess)
        return copy.deepcopy(sess)

    def claim(self, chat_id, seq):
        s = self.sessions.get(chat_id)
        if not s or s["seq"] != seq:
            return None
        s["seq"] += 1
        return copy.deepcopy(s)

    def save_session(self, sess):
        self.sessions[sess["chat_id"]] = copy.deepcopy(sess)


class FakeEffects:
    def __init__(self):
        self.records, self.leads, self.events = [], [], []
        self.rand = 0.0
        self.clock = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)

    def record(self, sess, entries, acted):
        self.records.append((sess["chat_id"], list(entries), acted))

    def upsert_lead(self, project_id, visitor_id, fields, custom):
        self.leads.append((visitor_id, fields, custom))
        return "lead-1"

    def log_events(self, events):
        self.events.extend(events)

    def now(self):
        return self.clock

    def today(self):
        return self.clock.date()

    def random(self):
        return self.rand


def node(nid, type_, content=None, start=False):
    return {"id": nid, "type": type_, "content": content or {}, "is_start": start}


def edge(a, trigger, b):
    return {"from_node_id": a, "trigger": trigger, "to_node_id": b}


def make(nodes, edges, free=False):
    store = FakeStore({"id": "flow-1", "free_questions": free}, nodes, edges)
    fx = FakeEffects()
    return Engine(store, fx, "https://app.example"), store, fx


# --------------------------------------------------------------------------- #
# Templating / conditions
# --------------------------------------------------------------------------- #
class TemplatingTests(unittest.TestCase):
    def test_fallback_and_single_pass(self):
        self.assertEqual(render_text("Hi {{name|there}}!", {}), "Hi there!")
        self.assertEqual(render_text("Hi {{name}}", {"name": "{{email}}", "email": "x@y.z"}), "Hi {{email}}")
        self.assertEqual(render_text("{{n}}", {"n": 5.0}), "5")

    def test_url_rules(self):
        self.assertEqual(render_url("https://a.com/x?q={{q}}", {"q": "a b&c"}), "https://a.com/x?q=a%20b%26c")
        with self.assertRaises(ValueError):
            render_url("https://{{host}}/x", {"host": "evil.com"})

    def test_set_variable_limits(self):
        v = {}
        self.assertTrue(set_variable(v, "budget", " 50 "))
        self.assertEqual(v["budget"], "50")
        self.assertFalse(set_variable(v, "page_url", "x"))
        self.assertFalse(set_variable(v, "Bad-Name", "x"))
        self.assertEqual(to_decimal("Rs 1,50,000"), 150000)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class EngineTests(unittest.TestCase):
    def basic(self, free=False):
        nodes = [
            node("n1", "message", {"body": "Welcome!"}, start=True),
            node("n2", "quick_replies", {"body": "Pick one", "options": [
                {"id": "o_a", "label": "Pricing"}, {"id": "o_b", "label": "Talk to us"}]}),
            node("n3", "message", {"body": "Pricing starts at 99"}),
            node("n4", "form", {"title": "Your details", "save_lead": True, "fields": [
                {"name": "name", "type": "text", "label": "Name"},
                {"name": "email", "type": "email", "label": "Email"},
                {"name": "city", "type": "text", "label": "City", "required": False}]}),
            node("n5", "message", {"body": "Thanks {{name}}!"}),
        ]
        edges = [edge("n1", "next", "n2"), edge("n2", "o_a", "n3"), edge("n2", "o_b", "n4"),
                 edge("n4", "next", "n5")]
        return make(nodes, edges, free)

    def test_start_runs_until_input(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        self.assertEqual([m["text"] for m in env["messages"]], ["Welcome!", "Pick one"])
        self.assertEqual(env["input"]["kind"], "choices")
        self.assertEqual(env["input"]["layout"], "chips")
        self.assertEqual(env["nodeId"], "n2")
        # nothing visitor-made yet -> transcript recorded as not-acted
        self.assertFalse(fx.records[-1][2])

    def test_choice_form_lead_and_template(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        sid = env["sessionId"]
        env = eng.step(PID, sid, VID, env["seq"], "n2", {"type": "choice", "id": "o_b"})
        self.assertEqual(env["input"]["kind"], "form")
        bad = eng.step(PID, sid, VID, env["seq"], "n4", {"type": "form", "values": {"name": "Asha", "email": "nope"}})
        self.assertEqual(bad["error"]["code"], "invalid")
        self.assertIn("email", bad["error"]["fields"])
        self.assertEqual(bad["nodeId"], "n4")
        ok = eng.step(PID, sid, VID, bad["seq"], "n4", {"type": "form", "values": {
            "name": "Asha <b>", "email": "A@X.com", "city": "Chennai"}})
        self.assertEqual(ok["messages"][-1]["text"], "Thanks Asha <b>!")   # plain text, widget escapes
        self.assertEqual(fx.leads[-1], (VID, {"name": "Asha <b>", "email": "a@x.com"}, {"city": "Chennai"}))

    def test_double_tap_is_resync_without_effects(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        sid, seq = env["sessionId"], env["seq"]
        first = eng.step(PID, sid, VID, seq, "n2", {"type": "choice", "id": "o_a"})
        records = len(fx.records)
        second = eng.step(PID, sid, VID, seq, "n2", {"type": "choice", "id": "o_a"})   # stale seq
        self.assertTrue(second.get("resync"))
        self.assertEqual(len(fx.records), records)
        self.assertEqual(second["messages"], [])
        self.assertEqual(first["messages"][0]["text"], "Pricing starts at 99")

    def test_wrong_visitor_or_project_is_expired(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        self.assertEqual(eng.step(PID, env["sessionId"], "someone-else", env["seq"], "n2",
                                  {"type": "choice", "id": "o_a"})["status"], "expired")

    def test_free_text_reprompt_or_ai(self):
        eng, store, fx = self.basic(free=False)
        env = eng.start(PID, None, VID, "open")
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "what is this?"})
        self.assertEqual(env["messages"][0]["text"], "Please choose one of the options above.")
        self.assertEqual(env["input"]["kind"], "choices")
        # typing an exact label counts as tapping it
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "pricing"})
        self.assertEqual(env["messages"][0]["text"], "Pricing starts at 99")

        eng, store, fx = self.basic(free=True)
        env = eng.start(PID, None, VID, "open")
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "what is this?"})
        self.assertEqual(env["delegate"], "ai")
        self.assertEqual(env["input"]["kind"], "choices")

    def test_end_of_flow_restart_or_ai(self):
        eng, store, fx = self.basic(free=False)
        env = eng.start(PID, None, VID, "open")
        env = eng.step(PID, env["sessionId"], VID, env["seq"], "n2", {"type": "choice", "id": "o_a"})
        self.assertEqual(env["status"], "ended")
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "hello"})
        self.assertEqual(env["messages"][0]["text"], "Welcome!")      # restarted

        eng, store, fx = self.basic(free=True)
        env = eng.start(PID, None, VID, "open")
        env = eng.step(PID, env["sessionId"], VID, env["seq"], "n2", {"type": "choice", "id": "o_a"})
        self.assertEqual(env["status"], "ai")
        self.assertTrue(env["menuChip"])
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "hello"})
        self.assertEqual(env["delegate"], "ai")

    def test_loop_guard(self):
        nodes = [node("a", "message", {"body": "A"}, start=True), node("b", "message", {"body": "B"})]
        eng, store, fx = make(nodes, [edge("a", "next", "b"), edge("b", "next", "a")])
        env = eng.start(PID, None, VID, "open")
        self.assertEqual(len(env["messages"]), MAX_AUTO_STEPS)
        self.assertTrue(any(e["event"] == "loop_guard" for e in fx.events))
        self.assertEqual(env["status"], "ended")

    def test_removed_node_types_end_quietly(self):
        # Condition / set variable / random split / webhook were removed; an
        # old saved flow with one must neither crash nor show its settings.
        nodes = [node("a", "message", {"body": "Hi"}, start=True),
                 node("w", "webhook", {"url": "https://hooks.example/secret-path",
                                       "body": [{"key": "k", "value": "v"}]})]
        eng, store, fx = make(nodes, [edge("a", "next", "w")])
        env = eng.start(PID, None, VID, "open")
        self.assertEqual([m["text"] for m in env["messages"]], ["Hi"])
        self.assertEqual(env["status"], "ended")
        self.assertNotIn("secret-path", json.dumps(env))

    def test_delays(self):
        nodes = [node("a", "message", {"body": "A"}, start=True),
                 node("d", "time_delay", {"delay_seconds": 3}),
                 node("b", "message", {"body": "B"}),
                 node("d2", "time_delay", {"delay_seconds": 2, "delay_unit": "minutes"}),
                 node("c", "message", {"body": "C"})]
        edges = [edge("a", "next", "d"), edge("d", "next", "b"), edge("b", "next", "d2"), edge("d2", "next", "c")]
        eng, store, fx = make(nodes, edges)
        env = eng.start(PID, None, VID, "open")
        kinds = [m["kind"] for m in env["messages"]]
        self.assertEqual(kinds, ["text", "pause", "text"])
        self.assertEqual(env["continueAfterMs"], 120000)
        early = eng.step(PID, env["sessionId"], VID, env["seq"], "d2", {"type": "continue"})
        self.assertEqual(early["messages"], [])
        self.assertGreater(early["continueAfterMs"], 0)
        fx.clock += timedelta(minutes=2)
        done = eng.step(PID, env["sessionId"], VID, early["seq"], "d2", {"type": "continue"})
        self.assertEqual(done["messages"][0]["text"], "C")

    def test_flow_switched_or_deactivated(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        store.flow = None
        out = eng.step(PID, env["sessionId"], VID, env["seq"], "n2", {"type": "choice", "id": "o_a"})
        self.assertEqual(out["status"], "inactive")

        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        store.flow = {"id": "flow-2", "free_questions": False}
        out = eng.step(PID, env["sessionId"], VID, env["seq"], "n2", {"type": "choice", "id": "o_a"})
        self.assertEqual(out["messages"][0]["text"], "This conversation was updated - let's start again.")

    def test_deleted_node_restarts(self):
        eng, store, fx = self.basic()
        env = eng.start(PID, None, VID, "open")
        del store.nodes["n2"]
        store.nodes["n1"]["is_start"] = True
        out = eng.step(PID, env["sessionId"], VID, env["seq"], "n2", {"type": "choice", "id": "o_a"})
        self.assertIn("start again", out["messages"][0]["text"])

    def test_output_nodes_are_safe(self):
        nodes = [node("a", "open_url", {"body": "x", "url": "javascript:alert(1)"}, start=True),
                 node("b", "message_media", {"body": "pic", "media_url": "http://insecure/x.png"}),
                 node("c", "call_us", {"phone": "+91 98765 43210"}),
                 node("d", "carousel", {"cards": [{"title": "T", "image": "https://x/i.png",
                                                   "buttons": [{"id": "c1", "label": "Choose"},
                                                               {"label": "Site", "url": "javascript:x"}]}]})]
        edges = [edge("a", "next", "b"), edge("b", "next", "c"), edge("c", "next", "d")]
        eng, store, fx = make(nodes, edges)
        env = eng.start(PID, None, VID, "open")
        self.assertEqual(env["messages"][0], {"kind": "text", "text": "x"})          # js: url dropped
        self.assertEqual(env["messages"][1], {"kind": "text", "text": "pic"})        # http media dropped
        self.assertEqual(env["messages"][2]["url"], "tel:+919876543210")
        self.assertEqual(env["input"]["cards"][0]["buttons"], [{"label": "Choose", "id": "c1"}])

    def test_human_mode_saves_text_without_reply(self):
        nodes = [node("h", "talk_to_human", {"body": "Connecting..."}, start=True)]
        eng, store, fx = make(nodes, [])
        env = eng.start(PID, None, VID, "open")
        self.assertEqual(env["status"], "human")
        out = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "hello?"})
        self.assertEqual(out["messages"], [])
        self.assertEqual(fx.records[-1][1], [{"role": "user", "content": "hello?"}])



class WhatsAppParityTests(unittest.TestCase):
    """Idle restart and the way back to the menu behave like WhatsApp flows."""
    def flow(self, free=True):
        nodes = [
            node("n1", "quick_replies", {"body": "Menu", "options": [
                {"id": "o_a", "label": "Hi"}, {"id": "o_b", "label": "Prices"}]}, start=True),
            node("n2", "ask_a_question", {"body": "Ask away"}),
        ]
        eng, store, fx = make(nodes, [edge("n1", "o_b", "n2")], free)
        return eng, store, fx

    def to_ai(self, eng):
        env = eng.start(PID, None, VID, "open")
        return eng.step(PID, env["sessionId"], VID, env["seq"], "n1", {"type": "choice", "id": "o_b"})

    def test_idle_two_hours_restarts_menu(self):
        eng, store, fx = self.flow()
        env = self.to_ai(eng)
        self.assertEqual(env["status"], "ai")
        sid = env["sessionId"]
        store.sessions[sid]["updated_at"] = (fx.clock - timedelta(hours=1)).isoformat()
        self.assertEqual(eng.step(PID, sid, VID, env["seq"], None, {"type": "text", "text": "price?"})["delegate"], "ai")
        env = eng.step(PID, sid, VID, store.sessions[sid]["seq"], None, {"type": "text", "text": "price?"})
        store.sessions[sid]["updated_at"] = (fx.clock - timedelta(hours=3)).isoformat()
        env = eng.step(PID, sid, VID, store.sessions[sid]["seq"], None, {"type": "text", "text": "price?"})
        self.assertEqual(env["messages"][0]["text"], "Menu")
        self.assertEqual(env["input"]["kind"], "choices")

    def test_opening_message_is_saved(self):
        eng, store, fx = self.flow()
        env = eng.start(PID, None, VID, "text", None, "do you deliver?")
        self.assertEqual(env["messages"][0]["text"], "Menu")
        chat, entries, acted = fx.records[-1]
        self.assertTrue(acted)
        self.assertEqual(entries[0], {"role": "user", "content": "do you deliver?"})

    def test_recent_team_reply_is_not_idle(self):
        eng, store, fx = self.flow()
        env = self.to_ai(eng)
        sid = env["sessionId"]
        store.sessions[sid].update(mode="human",
                                   updated_at=(fx.clock - timedelta(hours=5)).isoformat(),
                                   last_agent_msg_at=(fx.clock - timedelta(minutes=10)).isoformat())
        out = eng.step(PID, sid, VID, store.sessions[sid]["seq"], None, {"type": "text", "text": "thanks"})
        self.assertEqual(out["status"], "human")

    def test_question_offers_back_to_menu_instead_of_keywords(self):
        nodes = [
            node("n1", "quick_replies", {"body": "Menu", "options": [{"id": "o_a", "label": "Book"}]}, start=True),
            node("n2", "ask_input", {"body": "Your city?", "var": "city"}),
        ]
        eng, store, fx = make(nodes, [edge("n1", "o_a", "n2")])
        env = eng.start(PID, None, VID, "open")
        self.assertFalse(env["menuChip"])                      # buttons on screen: no extra chip
        env = eng.step(PID, env["sessionId"], VID, env["seq"], "n1", {"type": "choice", "id": "o_a"})
        self.assertEqual(env["input"]["kind"], "field")
        self.assertTrue(env["menuChip"])                       # the way out of a question
        # typing "menu" is just an answer now - no keywords
        env = eng.step(PID, env["sessionId"], VID, env["seq"], None, {"type": "text", "text": "menu"})
        self.assertEqual(store.sessions[env["sessionId"]]["variables"]["city"], "menu")
        # the chip itself restarts the flow
        sid = env["sessionId"]
        env = eng.step(PID, sid, VID, store.sessions[sid]["seq"], None, {"type": "menu"})
        self.assertEqual(env["messages"][0]["text"], "Menu")


class OptionIdGoldenTests(unittest.TestCase):
    def test_unchanged(self):
        gold = {"Veg Meals": "veg_meals", "Price?": "price", "": "opt_45h", "A": "a",
                "  Talk to a human ": "talk_to_a_human"}
        for label, expected in gold.items():
            self.assertEqual(option_id(label), expected)


if __name__ == "__main__":
    unittest.main()
