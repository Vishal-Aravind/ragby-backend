"""Custom template rules and the components sent to Meta. No network.

Run from backend/:  python -m unittest tests.test_wa_templates -v
"""
import unittest

from wa_templates import TemplateSpec, TemplateButton, build_components


def spec(**kw):
    base = dict(project_id="p", name="diwali_offer", category="MARKETING", language="en_US",
                body="Hi {{1}}, get {{2}}% off this Diwali.", body_examples=["Asha", "20"])
    base.update(kw)
    return TemplateSpec(**base)


class TemplateRulesTests(unittest.TestCase):
    def test_full_template_builds(self):
        s = spec(header_type="text", header_text="Offer for {{1}}", header_example="Asha",
                 footer="Reply STOP to opt out",
                 buttons=[TemplateButton(type="url", text="Shop now", url="https://shop.test"),
                          TemplateButton(type="quick_reply", text="Interested"),
                          TemplateButton(type="phone", text="Call us", phone="+91 98765 43210")])
        errors, comps = build_components(s)
        self.assertEqual(errors, [])
        types = [c["type"] for c in comps]
        self.assertEqual(types, ["HEADER", "BODY", "FOOTER", "BUTTONS"])
        self.assertEqual(comps[0]["example"], {"header_text": ["Asha"]})
        self.assertEqual(comps[1]["example"], {"body_text": [["Asha", "20"]]})
        # quick replies first, then links/calls; phone cleaned up
        self.assertEqual([b["type"] for b in comps[3]["buttons"]], ["QUICK_REPLY", "URL", "PHONE_NUMBER"])
        self.assertEqual(comps[3]["buttons"][2]["phone_number"], "+919876543210")

    def test_plain_body_no_examples(self):
        errors, comps = build_components(spec(body="Thanks for shopping with us.", body_examples=[]))
        self.assertEqual(errors, [])
        self.assertNotIn("example", comps[0])

    def test_name_and_choices(self):
        errors, _ = build_components(spec(name="Diwali Offer", category="AUTH", language="fr"))
        self.assertEqual(len(errors), 3)

    def test_variable_rules(self):
        cases = {
            "{{1}} welcome to our store": "start",
            "Welcome to our store {{1}}": "end",
            "Hi {{1}}{{2}} welcome": "next to each other",
            "Hi {{1}} and {{3}} welcome": "in order",
        }
        for body, phrase in cases.items():
            errors, _ = build_components(spec(body=body, body_examples=["a", "b", "c"]))
            self.assertTrue(any(phrase in e for e in errors), (body, errors))
        # a full stop after the last variable is fine
        errors, _ = build_components(spec(body="Hi there, your code is {{1}}.", body_examples=["X1"]))
        self.assertEqual(errors, [])

    def test_missing_examples(self):
        errors, _ = build_components(spec(body_examples=["Asha"]))
        self.assertTrue(any("example value" in e for e in errors))

    def test_header_rules(self):
        errors, _ = build_components(spec(header_type="text", header_text="Hi {{1}} and {{2}}", header_example="x"))
        self.assertTrue(any("only one variable" in e for e in errors))
        errors, _ = build_components(spec(header_type="image", header_image_url=None))
        self.assertTrue(any("upload an image" in e for e in errors))
        errors, comps = build_components(spec(header_type="image", header_image_url="https://x/y.png"), header_handle="H123")
        self.assertEqual(errors, [])
        self.assertEqual(comps[0], {"type": "HEADER", "format": "IMAGE", "example": {"header_handle": ["H123"]}})

    def test_footer_and_buttons(self):
        errors, _ = build_components(spec(footer="Hi {{1}}"))
        self.assertTrue(any("Footer" in e for e in errors))
        too_many = [TemplateButton(type="url", text=f"L{i}", url="https://a.test") for i in range(3)]
        errors, _ = build_components(spec(buttons=too_many))
        self.assertTrue(any("website buttons" in e for e in errors))
        errors, _ = build_components(spec(buttons=[TemplateButton(type="url", text="Go", url="https://a.test/{{1}}")]))
        self.assertTrue(any("variables aren't supported" in e for e in errors))
        errors, _ = build_components(spec(buttons=[TemplateButton(type="phone", text="Call", phone="98765")]))
        self.assertTrue(any("country code" in e for e in errors))
        errors, _ = build_components(spec(buttons=[TemplateButton(type="quick_reply", text="x" * 30)]))
        self.assertTrue(any("25 characters" in e for e in errors))


class CampaignHeaderTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from campaigns import _template_header
        self.check = _template_header
        self.req = lambda **kw: SimpleNamespace(**{"variables": [], "header_image_url": None, "header_text": None, **kw})

    def tpl(self, header=None, body="Hi {{1}}, welcome.", buttons=None):
        comps = []
        if header:
            comps.append(header)
        comps.append({"type": "BODY", "text": body})
        if buttons:
            comps.append({"type": "BUTTONS", "buttons": buttons})
        return {"name": "t", "components": comps}

    def test_image_header_needs_image(self):
        from fastapi import HTTPException
        t = self.tpl({"type": "HEADER", "format": "IMAGE"})
        with self.assertRaises(HTTPException):
            self.check(t, self.req(variables=["Asha"]))
        self.assertEqual(self.check(t, self.req(variables=["Asha"], header_image_url="https://x/a.png")), {"image": "https://x/a.png"})

    def test_text_header_variable(self):
        from fastapi import HTTPException
        t = self.tpl({"type": "HEADER", "format": "TEXT", "text": "Offer for {{1}}"})
        with self.assertRaises(HTTPException):
            self.check(t, self.req(variables=["Asha"]))
        self.assertEqual(self.check(t, self.req(variables=["Asha"], header_text="Asha")), {"text": "Asha"})
        plain = self.tpl({"type": "HEADER", "format": "TEXT", "text": "Big sale"})
        self.assertIsNone(self.check(plain, self.req(variables=["Asha"])))

    def test_unsupported_and_missing_values(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            self.check(self.tpl({"type": "HEADER", "format": "VIDEO"}), self.req(variables=["A"]))
        with self.assertRaises(HTTPException):
            self.check(self.tpl(buttons=[{"type": "URL", "text": "Go", "url": "https://a.test/{{1}}"}]), self.req(variables=["A"]))
        with self.assertRaises(HTTPException):
            self.check(self.tpl(), self.req(variables=[""]))
        self.assertIsNone(self.check(self.tpl(), self.req(variables=["A"])))

    def test_sender_puts_header_first(self):
        import campaigns
        sent = []
        class R:
            ok = True
            text = ""
            def json(self):
                return {}
        campaigns.http.post = lambda url, **k: sent.append(k["json"]) or R()
        campaigns.supabase.table = lambda *a, **k: type("Q", (), {"__getattr__": lambda s, n: (lambda *a, **k: s), "execute": lambda s: type("X", (), {"data": [{"status": "sending"}]})()})()
        campaigns.time = __import__("time")
        campaigns.send_campaign_messages("c1", [{"phone": "919999999999"}], "t", "en_US", ["Asha"], "PID", "TOK",
                                         {"image": "https://x/a.png"})
        comps = sent[0]["template"]["components"]
        self.assertEqual(comps[0], {"type": "header", "parameters": [{"type": "image", "image": {"link": "https://x/a.png"}}]})
        self.assertEqual(comps[1]["type"], "body")


if __name__ == "__main__":
    unittest.main()
