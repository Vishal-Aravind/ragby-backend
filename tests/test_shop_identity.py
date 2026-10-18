"""Shop customer tokens and receipts. No network, no database.

Run from backend/:  python -m unittest tests.test_shop_identity -v
"""
import base64
import json
import os
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

import shop_identity as si

PID = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
PHONE = "919876543210"


def _env(**kw):
    base = {"SHOP_LINK_SECRET": "x" * 40, "SHOP_TOKEN_TTL_DAYS": "7"}
    base.update(kw)
    return mock.patch.dict(os.environ, base)


class TokenTests(unittest.TestCase):
    def test_issue_and_verify(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE)
            self.assertTrue(t.startswith("v1."))
            self.assertEqual(len(t.split(".")), 3)
            self.assertNotIn("=", t)
            self.assertLess(len(t), 512)
            self.assertEqual(si.verify_customer_token(t, PID), (PHONE, None))
            # project ids compare case-insensitively
            self.assertEqual(si.verify_customer_token(t, PID.upper()), (PHONE, None))

    def test_bad_phone_not_issued(self):
        with _env():
            self.assertIsNone(si.issue_customer_token(PID, "12ab"))
            self.assertIsNone(si.issue_customer_token(PID, "1234567"))

    def test_expiry(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE, now=time.time() - 8 * 86400)
            self.assertEqual(si.verify_customer_token(t, PID), (None, "expired"))
            t2 = si.issue_customer_token(PID, PHONE, now=time.time() - 6 * 86400)
            self.assertEqual(si.verify_customer_token(t2, PID)[0], PHONE)

    def test_future_iat_rejected(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE, now=time.time() + 3600)
            self.assertEqual(si.verify_customer_token(t, PID), (None, "malformed"))

    def test_ttl_capped(self):
        with _env(SHOP_TOKEN_TTL_DAYS="30"):
            self.assertEqual(si.token_ttl_days(), 7)

    def test_tamper(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE)
            v, payload, sig = t.split(".")
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            data["ph"] = "919999999999"
            forged = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":"), sort_keys=True).encode()).decode().rstrip("=")
            self.assertEqual(si.verify_customer_token(f"v1.{forged}.{sig}", PID), (None, "bad_signature"))
            flipped = sig[:-1] + ("A" if sig[-1] != "A" else "B")
            self.assertEqual(si.verify_customer_token(f"v1.{payload}.{flipped}", PID)[1], "bad_signature")

    def test_malformed(self):
        with _env():
            for bad in ("abc", "v2.a.b", "v1..", "v1.a.b.c", "x" * 600):
                self.assertEqual(si.verify_customer_token(bad, PID)[0], None)
                self.assertIn(si.verify_customer_token(bad, PID)[1], ("malformed", "bad_signature"))
            self.assertEqual(si.verify_customer_token(None, PID), (None, "missing"))
            self.assertEqual(si.verify_customer_token("", PID), (None, "missing"))

    def test_wrong_project(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE)
            self.assertEqual(si.verify_customer_token(t, OTHER), (None, "wrong_project"))

    def test_secret_rotation_revokes(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE)
        with _env(SHOP_LINK_SECRET="y" * 40):
            self.assertEqual(si.verify_customer_token(t, PID), (None, "bad_signature"))

    def test_no_secret_fails_closed(self):
        with _env(SHOP_LINK_SECRET=""), mock.patch.object(si, "SUPABASE_SERVICE_ROLE_KEY", None):
            self.assertIsNone(si.issue_customer_token(PID, PHONE))
            self.assertEqual(si.verify_customer_token("v1.a.b", PID), (None, "unavailable"))
            self.assertIsNone(si.order_receipt("o1"))
            self.assertFalse(si.verify_receipt("o1", "anything"))

    def test_domain_separation(self):
        """A receipt key never equals the token key."""
        with _env():
            self.assertNotEqual(si._key(si.TOKEN_KEY_LABEL), si._key(si.RECEIPT_KEY_LABEL))


class IdentityTests(unittest.TestCase):
    def test_token_wins_over_body_phone(self):
        with _env():
            t = si.issue_customer_token(PID, PHONE)
            self.assertEqual(si.resolve_identity(PID, t, "918888888888"),
                             {"phone": PHONE, "verified": True, "reason": None})

    def test_legacy_phone_unverified(self):
        with _env():
            ident = si.resolve_identity(PID, None, "+91 98765-43210")
            self.assertEqual(ident, {"phone": PHONE, "verified": False, "reason": "missing"})
            self.assertIsNone(si.resolve_identity(PID, "garbage", "12")["phone"])

    def test_mask(self):
        self.assertEqual(si.mask_phone(PHONE), "+91 98•••••210")


class ReceiptAndUrlTests(unittest.TestCase):
    def test_receipt(self):
        with _env():
            r = si.order_receipt("abc-order")
            self.assertEqual(len(r), 22)
            self.assertTrue(si.verify_receipt("abc-order", r))
            self.assertFalse(si.verify_receipt("other-order", r))
            self.assertFalse(si.verify_receipt("abc-order", None))

    def test_build_shop_url(self):
        with _env():
            url = si.build_shop_url(PID, PHONE, catalog_id="c1", order_id=None, branch_id="b1", view="orders")
            q = parse_qs(urlparse(url).query)
            self.assertEqual(list(q.keys()), ["catalog", "phone", "branch", "view", "t"])
            self.assertEqual(si.verify_customer_token(q["t"][0], PID), (PHONE, None))
            plain = si.build_shop_url(PID)
            self.assertTrue(plain.endswith(f"/shop/{PID}"))
            self.assertNotIn("t=", si.build_shop_url(PID, None, catalog_id="c1"))

    def test_return_url(self):
        with _env():
            url = si.build_return_url(PID, "o1")
            q = parse_qs(urlparse(url).query)
            self.assertEqual(q["paid"], ["o1"])
            self.assertTrue(si.verify_receipt("o1", q["r"][0]))


if __name__ == "__main__":
    unittest.main()
