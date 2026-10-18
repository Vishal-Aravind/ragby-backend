"""Shop v3: offer maths, offer evaluation, price_order v2, place-order
idempotency and the redemption lifecycle. Supabase, Razorpay and WhatsApp
are all faked (tests/fake_supabase.py); nothing touches the network.

Run from backend/:  python -m unittest tests.test_shop_offers -v
"""
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

import shop
import shop_branches
import shop_identity
import shop_offers
import shop_rules as r
from tests.fake_supabase import FakeSupabase

PID = "11111111-1111-1111-1111-111111111111"
BID = "33333333-3333-3333-3333-333333333333"
BID2 = "44444444-4444-4444-4444-444444444444"
CAT = "55555555-5555-5555-5555-555555555555"
PROD = "66666666-6666-6666-6666-666666666666"
PHONE = "919876543210"
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def offer(**kw):
    base = dict(id=str(uuid.uuid4()), project_id=PID, title="Offer", description=None, kind="percent", value=10,
                max_discount=None, min_order_value=0, application="automatic", code=None, is_visible=True,
                starts_at=None, ends_at=None, is_active=True, archived_at=None, branch_ids=[], fulfilment="any",
                first_order_only=False, usage_limit=None, per_customer_limit=None, sort_order=0,
                created_at="2026-10-01T00:00:00+00:00")
    base.update(kw)
    return base


def ctx(**kw):
    base = dict(now=NOW, branch_id=BID, fulfilment="delivery", subtotal=500, delivery_fee=40, verified=True,
                used_total=0, used_by_customer=0, has_prior_order=False)
    base.update(kw)
    return base


# ─────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────

class CodeAndLabelTests(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(r.normalize_code(" welcome 10 "), "WELCOME10")
        self.assertEqual(r.normalize_code("flat50"), "FLAT50")
        for bad in ("abc", "_ABCD", "-ABCD", "A" * 21, "AB$CD", "", None):
            self.assertIsNone(r.normalize_code(bad))

    def test_labels(self):
        self.assertEqual(r.offer_label(offer(kind="percent", value=10, max_discount=100, min_order_value=199,
                                             first_order_only=True, application="code", code="WELCOME10")),
                         "10% off up to ₹100 on orders above ₹199 · first order · use WELCOME10")
        self.assertEqual(r.offer_label(offer(kind="flat", value=50)), "₹50 off · auto-applied")
        self.assertEqual(r.offer_label(offer(kind="free_delivery", value=0, application="code", code="SECRET9",
                                             is_visible=False)), "Free delivery")


class SavingTests(unittest.TestCase):
    def test_percent_cap(self):
        self.assertEqual(r.offer_saving(offer(value=10, max_discount=100), 1200, 40)["items_discount"], 100)
        self.assertEqual(r.offer_saving(offer(value=10, max_discount=100), 400, 40)["items_discount"], 40)
        self.assertEqual(r.offer_saving(offer(value=100), 230, 40)["items_discount"], 230)

    def test_flat_capped_at_subtotal(self):
        self.assertEqual(r.offer_saving(offer(kind="flat", value=50), 30, 0)["items_discount"], 30)
        self.assertEqual(r.offer_saving(offer(kind="flat", value=50), 400, 0)["items_discount"], 50)

    def test_free_delivery(self):
        s = r.offer_saving(offer(kind="free_delivery", value=0, fulfilment="delivery"), 400, 40)
        self.assertEqual(s, {"items_discount": 0.0, "delivery_discount": 40.0})


class EligibilityTests(unittest.TestCase):
    def test_order_of_checks(self):
        o = offer(is_active=False, ends_at="2026-01-01T00:00:00Z")
        self.assertEqual(r.offer_ineligibility(o, ctx()), "paused")
        self.assertEqual(r.offer_ineligibility(offer(starts_at="2026-11-01T00:00:00Z"), ctx()), "not_started")
        self.assertEqual(r.offer_ineligibility(offer(ends_at="2026-10-17T12:00:00Z"), ctx()), "expired")
        self.assertEqual(r.offer_ineligibility(offer(branch_ids=[BID2]), ctx()), "branch")
        self.assertIsNone(r.offer_ineligibility(offer(branch_ids=[BID, BID2]), ctx()))
        self.assertEqual(r.offer_ineligibility(offer(fulfilment="pickup"), ctx()), "fulfilment")
        self.assertEqual(r.offer_ineligibility(offer(first_order_only=True), ctx(verified=False)), "login_required")
        self.assertEqual(r.offer_ineligibility(offer(per_customer_limit=1), ctx(verified=False)), "login_required")
        self.assertEqual(r.offer_ineligibility(offer(min_order_value=600), ctx()), "min_order")
        self.assertEqual(r.offer_ineligibility(offer(usage_limit=5), ctx(used_total=5)), "limit_reached")
        self.assertEqual(r.offer_ineligibility(offer(per_customer_limit=1), ctx(used_by_customer=1)), "already_used")
        self.assertEqual(r.offer_ineligibility(offer(first_order_only=True), ctx(has_prior_order=True)), "not_first_order")
        fd = offer(kind="free_delivery", value=0, fulfilment="delivery")
        self.assertEqual(r.offer_ineligibility(fd, ctx(delivery_fee=0)), "no_saving")
        self.assertEqual(r.offer_ineligibility(fd, ctx(fulfilment="pickup")), "fulfilment")

    def test_min_order_uses_subtotal_before_discount(self):
        o = offer(kind="flat", value=50, min_order_value=299)
        self.assertIsNone(r.offer_ineligibility(o, ctx(subtotal=299)))
        self.assertEqual(r.offer_reason_message("min_order", offer(min_order_value=299), {"subtotal": 250}),
                         "Add ₹49 more to use this offer (min order ₹299).")

    def test_messages(self):
        self.assertEqual(r.offer_reason_message("fulfilment", offer(fulfilment="pickup")),
                         "This offer is only for pickup orders.")
        self.assertIn("Velachery", r.offer_reason_message("branch", ctx={"branch_name": "Velachery"}))
        self.assertTrue(r.offer_reason_message("not_started", offer(starts_at="2026-11-01T03:30:00Z"),
                                               {"timezone": "Asia/Kolkata"}).endswith("1 Nov, 9:00 AM."))


class ChooseTests(unittest.TestCase):
    def cand(self, saving, **kw):
        return {"offer": offer(**kw), "saving": saving, "items_discount": saving, "delivery_discount": 0}

    def test_best_auto(self):
        a, b = self.cand(50, title="A"), self.cand(80, title="B")
        chosen, status = r.choose_offer([a, b], None)
        self.assertIs(chosen, b)
        self.assertIsNone(status)

    def test_auto_tie_earliest_end_then_oldest(self):
        a = self.cand(50, ends_at=None, created_at="2026-01-01T00:00:00Z")
        b = self.cand(50, ends_at="2026-12-01T00:00:00Z", created_at="2026-05-01T00:00:00Z")
        self.assertIs(r.choose_offer([a, b], None)[0], b)
        c = self.cand(50, created_at="2026-02-01T00:00:00Z")
        self.assertIs(r.choose_offer([c, a], None)[0], a)

    def test_code_vs_auto(self):
        auto = self.cand(80, title="Weekend 10%")
        code = self.cand(50, application="code", code="FLAT50")
        self.assertEqual(r.choose_offer([auto], code), (auto, "not_better"))
        bigger = self.cand(90, application="code", code="BIG90")
        self.assertEqual(r.choose_offer([auto], bigger), (bigger, "applied"))

    def test_tie_goes_to_code(self):
        auto = self.cand(50)
        code = self.cand(50, application="code", code="FLAT50")
        self.assertEqual(r.choose_offer([auto], code), (code, "applied"))
        self.assertEqual(r.choose_offer([], code), (code, "applied"))


class TotalsTests(unittest.TestCase):
    def test_gst_on_discounted_value(self):
        t = r.apply_offer_totals(1200, 5, 40, 100, 0)
        self.assertEqual(t, {"taxable_subtotal": 1100, "gst_amount": 55.0, "total": 1195.0})

    def test_free_delivery_does_not_change_gst(self):
        t = r.apply_offer_totals(400, 5, 40, 0, 40)
        self.assertEqual(t["gst_amount"], 20.0)
        self.assertEqual(t["total"], 420.0)

    def test_floor_zero(self):
        self.assertEqual(r.apply_offer_totals(100, 0, 0, 100, 0)["total"], 0.0)

    def test_fingerprint(self):
        base = {"branch_id": BID, "items": [{"product_id": "p1", "quantity": 1}, {"product_id": "p1", "quantity": 1}],
                "fulfilment": "pickup", "name": "Asha", "coupon": "welcome10", "line1": "ignored for pickup"}
        same = {**base, "items": [{"product_id": "p1", "quantity": 2}], "coupon": "WELCOME10", "line1": None}
        self.assertEqual(r.checkout_fingerprint(base), r.checkout_fingerprint(same))
        self.assertRegex(r.checkout_fingerprint(base), r"^[0-9a-f]{64}$")
        self.assertNotEqual(r.checkout_fingerprint(base), r.checkout_fingerprint({**base, "note": "less spicy"}))

    def test_offer_state(self):
        self.assertEqual(r.offer_state(offer(), NOW), "active")
        self.assertEqual(r.offer_state(offer(is_active=False), NOW), "paused")
        self.assertEqual(r.offer_state(offer(starts_at="2026-11-01T00:00:00Z"), NOW), "scheduled")
        self.assertEqual(r.offer_state(offer(ends_at="2026-10-01T00:00:00Z"), NOW), "expired")
        self.assertEqual(r.offer_state(offer(usage_limit=2), NOW, 2), "limit_reached")
        self.assertEqual(r.offer_state(offer(archived_at="2026-10-01T00:00:00Z"), NOW), "archived")

    def test_media_url(self):
        base = "https://abc.supabase.co"
        ok = f"{base}/storage/v1/object/public/flow-media/{PID}/shop-branding/1729150000000.webp"
        self.assertTrue(r.valid_media_url(ok, base + "/"))
        for bad in (ok.replace(".webp", ".svg"), ok.replace("shop-branding", "other"),
                    ok.replace(base, "https://evil.example"), ok + "?x=1", "http://x"):
            self.assertFalse(r.valid_media_url(bad, base))


# ─────────────────────────────────────────────
# evaluate_offers (DB faked)
# ─────────────────────────────────────────────

class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.offers = []
        self.counts = {}
        self.blocked = False
        self.db = FakeSupabase(tables={"offers": self.offers}, rpcs={
            "shop_offer_counts": lambda p: [{"offer_id": i, "used_total": self.counts.get(i, 0), "used_by_customer": 0}
                                            for i in p["p_offer_ids"]],
            "shop_customer_has_prior_order": lambda p: False,
        })
        self.limited = []
        patches = [
            mock.patch.object(shop_offers, "supabase", self.db),
            mock.patch.object(shop_offers, "is_blocked", lambda *a: self.blocked),
            mock.patch.object(shop_offers, "is_rate_limited", lambda key, *a: self.limited.append(key) or False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def ev(self, code=None, subtotal=500, fulfilment="delivery", fee=40, verified=True, enabled=True, **kw):
        return shop_offers.evaluate_offers(PID, {"id": BID, "name": "Velachery"}, fulfilment, subtotal, fee,
                                           {"phone": PHONE, "verified": verified}, code, "1.2.3.4", NOW,
                                           enabled=enabled, **kw)

    def test_disabled(self):
        e = self.ev(code="WELCOME10", enabled=False)
        self.assertIsNone(e["offer"])
        self.assertEqual(e["coupon"]["reason"], "login_required")
        self.assertIsNone(self.ev(enabled=False)["coupon"])

    def test_auto_applies(self):
        self.offers.append(offer(kind="percent", value=10, max_discount=30))
        e = self.ev()
        self.assertEqual(e["items_discount"], 30)
        self.assertEqual(e["offer"]["saving"], 30)
        self.assertIsNone(e["coupon"])

    def test_code_not_better(self):
        self.offers.append(offer(title="Weekend 15%", value=15))
        self.offers.append(offer(kind="flat", value=20, application="code", code="FLAT20"))
        e = self.ev(code="flat20")
        self.assertEqual(e["items_discount"], 75)
        self.assertEqual(e["coupon"]["status"], "not_better")
        self.assertIn("Weekend 15%", e["coupon"]["message"])

    def test_code_applied(self):
        self.offers.append(offer(value=5))
        self.offers.append(offer(kind="flat", value=50, application="code", code="FLAT50"))
        e = self.ev(code="FLAT50")
        self.assertEqual((e["items_discount"], e["coupon"]["status"]), (50, "applied"))
        self.assertIsNone(e["offer"]["id"] and None)

    def test_unknown_code_counts_bad_try(self):
        e = self.ev(code="NOPE1234")
        self.assertEqual(e["coupon"]["reason"], "invalid_code")
        self.assertIn(f"coupon-bad:{PID}:1.2.3.4", self.limited)
        self.assertIn(f"coupon-bad-phone:{PID}:{PHONE}", self.limited)

    def test_blocked_gets_rate_limited_for_every_code(self):
        self.offers.append(offer(kind="flat", value=50, application="code", code="FLAT50"))
        self.blocked = True
        e = self.ev(code="FLAT50")
        self.assertEqual(e["coupon"]["reason"], "rate_limited")
        self.assertEqual(e["items_discount"], 0)

    def test_free_delivery_only_for_delivery(self):
        self.offers.append(offer(kind="free_delivery", value=0, fulfilment="delivery"))
        self.assertEqual(self.ev()["delivery_discount"], 40)
        e = self.ev(fulfilment="pickup", fee=0)
        self.assertIsNone(e["offer"])
        self.assertEqual(e["available_offers"][0]["reason"], "fulfilment")

    def test_hidden_code_listed_only_when_typed(self):
        self.offers.append(offer(kind="flat", value=50, application="code", code="SECRET77", is_visible=False))
        self.assertEqual(self.ev()["available_offers"], [])
        e = self.ev(code="secret77")
        self.assertEqual(e["available_offers"][0]["code"], "SECRET77")

    def test_limit_and_exclusion(self):
        o = offer(kind="flat", value=50, application="code", code="FLAT50", usage_limit=1)
        self.offers.append(o)
        self.counts[o["id"]] = 1
        e = self.ev(code="FLAT50")
        self.assertEqual((e["coupon"]["status"], e["coupon"]["reason"]), ("rejected", "limit_reached"))
        e2 = self.ev(code="FLAT50", exclude_offer_ids=(o["id"],))
        self.assertEqual(e2["coupon"]["reason"], "limit_reached")
        self.assertNotIn(f"coupon-bad:{PID}:1.2.3.4", self.limited)

    def test_date_window(self):
        self.offers.append(offer(ends_at="2026-10-17T11:00:00Z"))
        self.offers.append(offer(starts_at="2026-10-18T00:00:00Z"))
        e = self.ev()
        self.assertIsNone(e["offer"])
        self.assertEqual(sorted(a["reason"] for a in e["available_offers"]), ["expired", "not_started"])


# ─────────────────────────────────────────────
# price_order v2 + place-order (DB faked)
# ─────────────────────────────────────────────

def _shop_db(online=True, offers=None, price=115.0):
    tables = {
        "shop_config": [{"project_id": PID, "is_enabled": True, "gst_percent": 5, "currency": "₹",
                         "currency_code": "INR", "razorpay_key_id": "rzp_k" if online else None,
                         "razorpay_key_secret": "s" if online else None}],
        "shop_branches": [{"id": BID, "project_id": PID, "name": "Velachery", "is_default": True, "is_active": True,
                           "accepting_orders": True, "pickup_enabled": True, "delivery_enabled": False,
                           "hours": None, "timezone": "Asia/Kolkata", "min_order_value": 0,
                           "free_delivery_above": None, "pincodes": [], "max_delivery_km": None}],
        "catalogs": [{"id": CAT, "project_id": PID, "branch_id": BID, "is_active": True}],
        "products": [{"id": PROD, "project_id": PID, "catalog_id": CAT, "name": "Ghee Pongal", "price": price,
                      "is_available": True, "image_url": None, "shopify_variant_id": None}],
        "orders": [],
        "offers": offers or [],
        "offer_redemptions": [],
        "razorpay_connections": [],
        "whatsapp_integrations": [],
    }
    state = {"claims": []}

    def claim(p):
        state["claims"].append(p)
        return state.get("claim_result", {"ok": True, "redemption_id": "r1"})

    rpcs = {
        "shop_offer_counts": lambda p: [],
        "shop_customer_has_prior_order": lambda p: False,
        "shop_claim_offer": claim,
        "shop_confirm_offer": lambda p: None,
        "shop_release_offer": lambda p: None,
    }
    db = FakeSupabase(tables=tables, rpcs=rpcs, unique={"orders": [("project_id", "client_ref")]},
                      defaults={"orders": {"payment_link_ids": [], "placed_at": None, "payment_id": None}})
    return db, state


class ShopHarness(unittest.TestCase):
    online = True
    offers = None

    def setUp(self):
        self.db, self.state = _shop_db(online=self.online, offers=self.offers)
        self.links = []
        self.sent = []

        class Resp:
            def __init__(self, n):
                self.status_code, self._n, self.text = 200, n, ""

            def json(self):
                return {"id": f"plink_{self._n}", "short_url": f"https://rzp.io/i/{self._n}"}

        def fake_rzp(method, path, project_id, **kw):
            if path == "/payment_links":
                self.links.append(kw.get("json"))
                return Resp(len(self.links))
            return Resp(0)

        class FakeRazorpayClient:
            def __init__(inner, auth):
                inner.payment_link = SimpleNamespace(
                    create=lambda payload: (self.links.append(payload) or
                                            {"id": f"plink_{len(self.links)}", "short_url": f"https://rzp.io/i/{len(self.links)}"}),
                    cancel=lambda link_id: None)

        import razorpay_oauth
        patches = [
            mock.patch.dict(os.environ, {"SHOP_LINK_SECRET": "z" * 40, "SHOP_REQUIRE_CUSTOMER_TOKEN": ""}),
            mock.patch.object(shop, "supabase", self.db),
            mock.patch.object(shop_branches, "supabase", self.db),
            mock.patch.object(shop_offers, "supabase", self.db),
            mock.patch.object(shop, "is_rate_limited", lambda *a, **k: False),
            mock.patch.object(shop_offers, "is_rate_limited", lambda *a, **k: False),
            mock.patch.object(shop_offers, "is_blocked", lambda *a, **k: False),
            mock.patch("phones.default_region_for_project", lambda pid: "IN"),
            mock.patch.object(razorpay_oauth, "_razorpay_api_request",
                              mock.Mock(side_effect=ValueError("no oauth"))),
            mock.patch("razorpay.Client", FakeRazorpayClient),
            mock.patch.object(shop, "_set_session", lambda *a: None),
            mock.patch.object(shop, "send_order_placed_message", lambda *a: self.sent.append("placed") or "text"),
            mock.patch.object(shop, "send_owner_order_alert", lambda *a, **k: self.sent.append("owner")),
            mock.patch.object(shop, "_send_cart_confirmation",
                              lambda *a, **k: self.sent.append("cart") or {"status": "ok"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.request = SimpleNamespace(headers={}, client=None)

    def body(self, **kw):
        base = dict(customer_token=shop_identity.issue_customer_token(PID, PHONE), idempotency_key=str(uuid.uuid4()),
                    branch_id=BID, items=[{"product_id": PROD, "quantity": 2}], fulfilment="pickup",
                    customer_name="Asha", expected_total=241.5, sold_out_preference="call_me", note="Less spicy")
        base.update(kw)
        return shop.PlaceOrderBody(**base)

    def place(self, body):
        return shop.place_order(PID, body, self.request)

    def place_err(self, body):
        with self.assertRaises(HTTPException) as cm:
            self.place(body)
        return cm.exception


class PriceOrderTests(ShopHarness):
    offers = [offer(kind="percent", value=10, max_discount=100)]

    def test_v1_callers_get_no_discount(self):
        branch = self.db.tables["shop_branches"][0]
        p = shop_branches.price_order(PID, branch, [shop.CartItem(product_id=PROD, quantity=2)], "pickup")
        self.assertEqual((p["subtotal"], p["gst_amount"], p["total"], p["discount_amount"]), (230, 11.5, 241.5, 0))
        self.assertIsNone(p["offer"])

    def test_v2_discount_and_gst(self):
        branch = self.db.tables["shop_branches"][0]
        p = shop_branches.price_order(PID, branch, [shop.CartItem(product_id=PROD, quantity=2)], "pickup",
                                      offers_enabled=True, identity={"phone": PHONE, "verified": True}, now_utc=NOW)
        self.assertEqual(p["items_discount"], 23)
        self.assertEqual(p["taxable_subtotal"], 207)
        self.assertEqual(p["gst_amount"], 10.35)
        self.assertEqual(p["total"], 217.35)
        self.assertEqual(p["errors"], [])


class PlaceOrderOnlineTests(ShopHarness):
    def test_same_key_same_order_same_link(self):
        b = self.body()
        first = self.place(b)
        self.assertEqual(first["state"], "payment_required")
        self.assertEqual(first["payment_url"], "https://rzp.io/i/1")
        self.assertTrue(shop_identity.verify_receipt(first["order_id"], first["receipt"]))
        again = self.place(b)
        self.assertEqual(again["order_id"], first["order_id"])
        self.assertEqual(again["payment_url"], first["payment_url"])
        self.assertEqual(len(self.links), 1)
        self.assertEqual(len(self.db.tables["orders"]), 1)
        o = self.db.tables["orders"][0]
        self.assertEqual((o["payment_status"], o["payment_method"], o["sold_out_preference"], o["special_request"]),
                         ("link_sent", "online", "call_me", "Less spicy"))
        self.assertEqual(o["payment_link_ids"], ["plink_1"])
        self.assertIsNotNone(o["placed_at"])
        self.assertEqual(self.links[0]["callback_method"], "get")
        self.assertEqual(self.links[0]["amount"], 24150)

    def test_changed_cart_while_paying(self):
        b = self.body()
        self.place(b)
        e = self.place_err(self.body(idempotency_key=b.idempotency_key, note="Extra spicy"))
        self.assertEqual(e.detail["code"], "payment_in_progress")
        self.assertEqual(e.detail["payment_url"], "https://rzp.io/i/1")

    def test_expired_link_regenerated_on_same_order(self):
        b = self.body()
        first = self.place(b)
        o = self.db.tables["orders"][0]
        o["payment_link_expires_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        again = self.place(b)
        self.assertEqual(again["order_id"], first["order_id"])
        self.assertEqual(again["payment_url"], "https://rzp.io/i/2")
        self.assertEqual(o["payment_link_ids"], ["plink_1", "plink_2"])
        self.assertEqual(o["payment_id"], "plink_2")

    def test_other_phone_same_key(self):
        b = self.body()
        self.place(b)
        other = shop_identity.issue_customer_token(PID, "918888888888")
        e = self.place_err(self.body(idempotency_key=b.idempotency_key, customer_token=other))
        self.assertEqual(e.detail["code"], "idempotency_conflict")

    def test_price_changed(self):
        e = self.place_err(self.body(expected_total=1.0))
        self.assertEqual(e.detail["code"], "price_changed")
        self.assertEqual(e.detail["quote"]["total"], 241.5)
        self.assertEqual(self.db.tables["orders"], [])

    def test_paid_retry_is_already_placed(self):
        b = self.body()
        self.place(b)
        self.db.tables["orders"][0].update(status="confirmed", payment_status="paid")
        self.assertEqual(self.place(b)["state"], "already_placed")
        self.assertEqual(len(self.links), 1)

    def test_validation(self):
        self.assertEqual(self.place_err(self.body(idempotency_key="nope")).detail["code"], "invalid_request")
        self.assertEqual(self.place_err(self.body(customer_name="  ")).detail["code"], "invalid_request")
        self.assertEqual(self.place_err(self.body(note="x" * 301)).detail["code"], "invalid_request")
        e = self.place_err(self.body(fulfilment="delivery"))
        self.assertEqual(e.detail["code"], "invalid_request")


class PlaceOrderOfflineTests(ShopHarness):
    online = False

    def test_verified_offline_placed_once(self):
        self.db.tables["whatsapp_integrations"].append(
            {"project_id": PID, "phone_number_id": "pn1", "access_token": "tok"})
        b = self.body()
        res = self.place(b)
        self.assertEqual((res["state"], res["payment_method"]), ("placed", "offline"))
        o = self.db.tables["orders"][0]
        self.assertEqual((o["status"], o["payment_status"]), ("confirmed", "unpaid"))
        self.assertEqual(self.sent, ["placed", "owner"])
        again = self.place(b)
        self.assertEqual(again["state"], "already_placed")
        self.assertEqual(self.sent, ["placed", "owner"])       # never double-messages

    def test_unverified_offline_confirms_in_whatsapp(self):
        res = self.place(self.body(customer_token=None, phone="+91 98765 43210"))
        self.assertEqual(res["state"], "confirm_in_whatsapp")
        self.assertEqual(self.sent, ["cart"])
        self.assertEqual(self.db.tables["orders"][0]["status"], "pending")

    def test_require_token(self):
        with mock.patch.dict(os.environ, {"SHOP_REQUIRE_CUSTOMER_TOKEN": "true"}):
            e = self.place_err(self.body(customer_token=None, phone=PHONE))
        self.assertEqual(e.detail["code"], "token_required")


class PlaceOrderOfferTests(ShopHarness):
    code_offer = offer(kind="flat", value=50, application="code", code="FLAT50")
    offers = [code_offer]

    def test_claim_held_for_online(self):
        res = self.place(self.body(coupon_code="flat50", expected_total=189.0))
        self.assertEqual(res["totals"]["items_discount"], 50)
        self.assertEqual(res["offer"]["code"], "FLAT50")
        claim = self.state["claims"][0]
        self.assertEqual((claim["p_offer_id"], claim["p_phone"], claim["p_verified"]), (self.code_offer["id"], PHONE, True))
        self.assertIsNotNone(claim["p_hold_until"])
        o = self.db.tables["orders"][0]
        self.assertEqual((o["discount_amount"], o["offer_code"], o["discount_type"]), (50, "FLAT50", "flat"))

    def test_claim_lost_reprices(self):
        self.state["claim_result"] = {"ok": False, "reason": "limit_reached"}
        e = self.place_err(self.body(coupon_code="FLAT50", expected_total=189.0))
        self.assertEqual((e.detail["code"], e.detail["reason"]), ("offer_unavailable", "limit_reached"))
        self.assertEqual(e.detail["quote"]["total"], 241.5)
        o = self.db.tables["orders"][0]
        self.assertEqual((o["total"], o["offer_id"], o["discount_amount"]), (241.5, None, 0))
        self.assertEqual(self.links, [])

    def test_bad_coupon(self):
        e = self.place_err(self.body(coupon_code="NOTREAL1"))
        self.assertEqual((e.detail["code"], e.detail["reason"]), ("coupon_invalid", "invalid_code"))


class LifecycleTests(unittest.TestCase):
    def test_rpc_helpers(self):
        db = FakeSupabase(rpcs={"shop_claim_offer": lambda p: {"ok": True, "redemption_id": "r"},
                                "shop_confirm_offer": lambda p: None, "shop_release_offer": lambda p: None})
        with mock.patch.object(shop, "supabase", db):
            self.assertTrue(shop.claim_offer("o1", "f1", PHONE, True, None, 23.456)["ok"])
            shop.confirm_offer("o1")
            shop.release_offer("o1")
        names = [c[0] for c in db.rpc_calls]
        self.assertEqual(names, ["shop_claim_offer", "shop_confirm_offer", "shop_release_offer"])
        self.assertEqual(db.rpc_calls[0][1]["p_discount"], 23.46)
        self.assertIsNone(db.rpc_calls[0][1]["p_hold_until"])
        self.assertEqual(db.rpc_calls[1][1], {"p_order_id": "o1"})

    def test_claim_failure_is_not_ok(self):
        db = FakeSupabase(rpcs={})   # rpc raises
        with mock.patch.object(shop, "supabase", db), mock.patch("sentry_sdk.capture_exception"):
            self.assertFalse(shop.claim_offer("o1", "f1", PHONE, True, None, 1)["ok"])
            shop.release_offer("o1")      # best-effort, never raises

    def test_status_label_and_money_lines(self):
        o = {"status": "pending", "payment_status": "link_sent", "fulfilment": "delivery"}
        self.assertEqual(shop.order_status_label(o), "Awaiting payment")
        self.assertEqual(shop.order_status_label({**o, "status": "completed"}), "Delivered")
        self.assertEqual(shop.order_status_label({"status": "ready"}), "Ready for pickup")
        lines = shop.order_money_lines({"subtotal": 230, "items_discount": 23, "offer_code": "WELCOME10",
                                        "gst_amount": 10.35, "fulfilment": "delivery", "delivery_fee": 40,
                                        "delivery_discount": 40}, "₹")
        self.assertEqual(lines.split("\n"), ["Item total: ₹230.00", "Discount (WELCOME10): −₹23.00", "GST: ₹10.35",
                                             "Delivery: ₹40.00", "Free delivery offer: −₹40.00"])


if __name__ == "__main__":
    unittest.main()
