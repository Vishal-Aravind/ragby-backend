"""Shop branch rules: hours, open state, delivery quote, status messages.
No network, no database.

Run from backend/:  python -m unittest tests.test_shop_rules -v
"""
import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import shop_rules as r

IST = ZoneInfo("Asia/Kolkata")


def ist(y, mo, d, h, mi=0):
    """A UTC instant for an IST wall-clock time."""
    return datetime(y, mo, d, h, mi, tzinfo=IST).astimezone(timezone.utc)


def branch(**kw):
    base = dict(id="b1", name="Velachery", timezone="Asia/Kolkata", hours=None, is_active=True,
                accepting_orders=True, pickup_enabled=True, delivery_enabled=True,
                lat=12.9791, lng=80.2209, pincodes=[], max_delivery_km=None,
                delivery_per_km=0, delivery_base_fee=0, free_delivery_above=None)
    base.update(kw)
    return base


DAY = {"closed": False, "open": "09:00", "close": "22:00"}
WEEK = {k: dict(DAY) for k in r.DAY_KEYS}
# 2026-10-16 is a Friday, 2026-10-17 Saturday, 2026-10-18 Sunday, 2026-10-19 Monday.


class HoursValidationTests(unittest.TestCase):
    def test_valid_and_null(self):
        self.assertIsNone(r.validate_hours(None))
        self.assertEqual(r.validate_hours({"mon": DAY}), {"mon": DAY})

    def test_rejects_bad_shapes(self):
        for bad in ({"monday": DAY}, {"mon": {"closed": False, "open": "24:00", "close": "02:00"}},
                    {"mon": {"closed": False, "open": "9:00", "close": "22:00"}},
                    {"mon": {"closed": "no", "open": "09:00", "close": "22:00"}},
                    {"mon": {**DAY, "extra": 1}}, ["mon"]):
            with self.assertRaises(ValueError):
                r.validate_hours(bad)

    def test_timezone(self):
        self.assertEqual(r.validate_timezone("Asia/Dubai"), "Asia/Dubai")
        with self.assertRaises(ValueError):
            r.validate_timezone("Mars/Base")


class OpenStatusTests(unittest.TestCase):
    def test_null_hours_always_open(self):
        s = r.branch_status(branch(), True, ist(2026, 10, 17, 3))
        self.assertTrue(s["is_open"])
        self.assertTrue(s["can_order"])
        self.assertEqual(s["label"], "Open now")

    def test_regular_open_and_closed(self):
        b = branch(hours=WEEK)
        s = r.branch_status(b, True, ist(2026, 10, 17, 12))
        self.assertEqual(s["label"], "Open now · closes 10:00 PM")
        self.assertEqual(s["closes_at_label"], "10:00 PM")
        self.assertEqual(s["today_hours_label"], "Today 9:00 AM – 10:00 PM")

        s = r.branch_status(b, True, ist(2026, 10, 17, 7))
        self.assertFalse(s["can_order"])
        self.assertEqual(s["reason"], "closed")
        self.assertEqual(s["label"], "Closed now · opens at 9:00 AM")

        s = r.branch_status(b, True, ist(2026, 10, 17, 23))
        self.assertEqual(s["label"], "Closed now · opens tomorrow at 9:00 AM")
        self.assertEqual(s["opens_at_label"], "tomorrow at 9:00 AM")

    def test_overnight(self):
        hours = {"fri": {"closed": False, "open": "18:00", "close": "02:00"}}
        b = branch(hours=hours)
        # Saturday 01:00 is still Friday's session.
        s = r.branch_status(b, True, ist(2026, 10, 17, 1))
        self.assertTrue(s["is_open"])
        self.assertEqual(s["closes_at_label"], "2:00 AM")
        # Friday 23:00 open, closing tomorrow 2 AM.
        self.assertTrue(r.branch_status(b, True, ist(2026, 10, 16, 23))["is_open"])
        # Saturday 03:00 closed; next opening is next Friday.
        s = r.branch_status(b, True, ist(2026, 10, 17, 3))
        self.assertFalse(s["is_open"])
        self.assertEqual(s["label"], "Closed now · opens Fri at 6:00 PM")

    def test_overnight_reopens_same_day(self):
        hours = {k: {"closed": False, "open": "18:00", "close": "02:00"} for k in r.DAY_KEYS}
        s = r.branch_status(branch(hours=hours), True, ist(2026, 10, 17, 3))
        self.assertEqual(s["label"], "Closed now · opens at 6:00 PM")

    def test_closed_day_and_missing_day(self):
        hours = {"sat": {"closed": True, "open": "09:00", "close": "22:00"},
                 "mon": dict(DAY)}
        s = r.branch_status(branch(hours=hours), True, ist(2026, 10, 17, 12))
        self.assertFalse(s["is_open"])
        self.assertEqual(s["today_hours_label"], "Closed today")
        # Sunday has no key = closed, so Monday is next.
        self.assertEqual(s["label"], "Closed now · opens Mon at 9:00 AM")

    def test_no_open_day(self):
        s = r.branch_status(branch(hours={}), True, ist(2026, 10, 17, 12))
        self.assertEqual(s["label"], "Closed")
        self.assertIsNone(s["next_open_at"])

    def test_24h(self):
        hours = {"sat": {"closed": False, "open": "00:00", "close": "00:00"}}
        s = r.branch_status(branch(hours=hours), True, ist(2026, 10, 17, 3))
        self.assertTrue(s["is_open"])
        self.assertEqual(s["label"], "Open 24 hours")
        self.assertEqual(s["today_hours_label"], "Open 24 hours")

    def test_timezone_is_used(self):
        # 12:00 UTC is 16:00 in Dubai (open) but 21:00 in Tokyo... and
        # 12:00 UTC is 05:00 in Los Angeles (closed).
        now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
        self.assertTrue(r.branch_status(branch(hours=WEEK, timezone="Asia/Dubai"), True, now)["is_open"])
        s = r.branch_status(branch(hours=WEEK, timezone="America/Los_Angeles"), True, now)
        self.assertFalse(s["is_open"])
        self.assertEqual(s["timezone"], "America/Los_Angeles")

    def test_bad_timezone_falls_back(self):
        s = r.branch_status(branch(timezone="Nope/Nowhere"), True, ist(2026, 10, 17, 12))
        self.assertEqual(s["timezone"], "Asia/Kolkata")

    def test_paused_and_inactive(self):
        s = r.branch_status(branch(accepting_orders=False), True, ist(2026, 10, 17, 12))
        self.assertEqual((s["reason"], s["label"]), ("paused", "Not taking orders right now"))
        self.assertFalse(s["can_order"])
        # Closed beats paused.
        s = r.branch_status(branch(hours=WEEK, accepting_orders=False), True, ist(2026, 10, 17, 7))
        self.assertEqual(s["reason"], "closed")
        self.assertEqual(r.branch_status(branch(is_active=False), True, ist(2026, 10, 17, 12))["reason"], "inactive")
        self.assertEqual(r.branch_status(branch(), False, ist(2026, 10, 17, 12))["reason"], "inactive")

    def test_format_time(self):
        self.assertEqual(r.format_time(0, 5), "12:05 AM")
        self.assertEqual(r.format_time(12, 0), "12:00 PM")
        self.assertEqual(r.format_time(21, 30), "9:30 PM")


class DeliveryQuoteTests(unittest.TestCase):
    # ~2.1 km north of the branch.
    NEAR = (12.9791 + 2.1 / 111.2, 80.2209)
    FAR = (12.9791 + 8 / 111.2, 80.2209)

    def test_neither_pincodes_nor_radius_is_off(self):
        q = r.quote_delivery(branch(), 100, *self.NEAR, "600042")
        self.assertEqual((q["available"], q["reason"]), (False, "delivery_off"))
        q = r.quote_delivery(branch(delivery_enabled=False, max_delivery_km=6), 100, *self.NEAR, None)
        self.assertEqual(q["reason"], "delivery_off")

    def test_location_required(self):
        q = r.quote_delivery(branch(max_delivery_km=6), 100, None, None, "600042")
        self.assertEqual(q["reason"], "location_required")

    def test_radius_fee_and_rounding(self):
        b = branch(max_delivery_km=6, delivery_base_fee=20, delivery_per_km=10)
        q = r.quote_delivery(b, 100, *self.NEAR, None)
        self.assertTrue(q["available"])
        self.assertAlmostEqual(q["distance_km"], 2.1, places=1)
        self.assertEqual(q["fee"], 41)   # ceil(20 + 10 * 2.1), no float-noise bump to 42
        self.assertIsInstance(q["fee"], int)

    def test_ceil_up(self):
        b = branch(max_delivery_km=6, delivery_base_fee=0, delivery_per_km=7, lat=0.0001, lng=10)
        q = r.quote_delivery(b, 100, 0.0001 + 1.234 / 111.195, 10, None)
        self.assertEqual(q["fee"], 9)    # 7 * 1.23 = 8.61 -> 9

    def test_out_of_area_and_pincode_override(self):
        b = branch(max_delivery_km=6, delivery_base_fee=20, delivery_per_km=10)
        q = r.quote_delivery(b, 100, *self.FAR, "600091")
        self.assertEqual(q["reason"], "out_of_area")
        b["pincodes"] = ["600091"]
        q = r.quote_delivery(b, 100, *self.FAR, "600091")
        self.assertTrue(q["available"])
        self.assertEqual(q["fee"], 100)  # fee still uses the real 8 km

    def test_pincode_only_branch(self):
        b = branch(pincodes=["600042"], lat=None, lng=None, delivery_base_fee=30)
        q = r.quote_delivery(b, 100, *self.NEAR, "600042")
        self.assertEqual((q["available"], q["fee"], q["distance_km"]), (True, 30, None))
        q = r.quote_delivery(b, 100, *self.NEAR, "600001")
        self.assertEqual(q["reason"], "out_of_area")
        b["delivery_per_km"] = 5
        q = r.quote_delivery(b, 100, *self.NEAR, "600042")
        self.assertEqual(q["reason"], "location_required")

    def test_free_above(self):
        b = branch(max_delivery_km=6, delivery_base_fee=20, delivery_per_km=10, free_delivery_above=499)
        self.assertEqual(r.quote_delivery(b, 499, *self.NEAR, None)["fee"], 0)
        self.assertEqual(r.quote_delivery(b, 498.99, *self.NEAR, None)["fee"], 41)

    def test_haversine(self):
        # Chennai Central to Velachery, roughly 12 km.
        self.assertAlmostEqual(r.haversine_km(13.0827, 80.2707, 12.9791, 80.2209), 12.7, delta=0.5)

    def test_delivery_available(self):
        self.assertFalse(r.delivery_available(branch()))
        self.assertTrue(r.delivery_available(branch(pincodes=["600042"])))
        self.assertFalse(r.delivery_available(branch(pincodes=["600042"], delivery_enabled=False)))


class InputTests(unittest.TestCase):
    def test_pincode(self):
        self.assertEqual(r.normalize_pincode(" 600042 ", "IN"), "600042")
        self.assertEqual(r.normalize_pincode(600042, "IN"), "600042")
        self.assertIsNone(r.normalize_pincode("060004", "IN"))
        self.assertIsNone(r.normalize_pincode("60004", "IN"))
        self.assertEqual(r.normalize_pincode("sw1a 1aa", "GB"), "SW1A 1AA")
        self.assertIsNone(r.normalize_pincode("!!", "GB"))

    def test_latlng(self):
        self.assertEqual(r.validate_latlng("12.12345678", 80.1), (12.123457, 80.1))
        for bad in ((0, 0), (91, 0), (0, 181), (float("nan"), 1), ("x", 1)):
            with self.assertRaises(ValueError):
                r.validate_latlng(*bad)

    def test_clean_text(self):
        self.assertEqual(r.clean_text("  a\x00b\n\n\n\nc  ", 80), "ab\n\nc")
        self.assertIsNone(r.clean_text("   ", 80))
        with self.assertRaises(ValueError):
            r.clean_text("x" * 81, 80)

    def test_money(self):
        self.assertEqual(r.money(230), "230")
        self.assertEqual(r.money(11.5), "11.50")


class StatusMessageTests(unittest.TestCase):
    CTX = {"name": "Asha", "oid": "A1B2C3D4", "branch": "Velachery", "contact": "+91 98765 43210",
           "prep": 20, "address": "12, 100 Feet Rd"}

    def msg(self, status, fulfilment="pickup", **ctx):
        return r.order_status_message(status, fulfilment, {**self.CTX, **ctx})

    def test_mapping(self):
        cases = [
            ("confirmed", "pickup", "Confirmed"),
            ("preparing", "delivery", "Being prepared"),
            ("ready", "pickup", "Ready for pickup"),
            ("ready", "delivery", "Ready, dispatching soon"),
            ("out_for_delivery", "delivery", "Out for delivery"),
            ("completed", "pickup", "Picked up"),
            ("completed", "delivery", "Delivered"),
        ]
        for status, fulfilment, text in cases:
            m = self.msg(status, fulfilment)
            self.assertEqual(m["template"], "order_status_update", status)
            self.assertEqual(m["params"], ["Asha", "A1B2C3D4", "Velachery", text])
            self.assertIn("A1B2C3D4", m["text"])

    def test_texts(self):
        self.assertIn("Ready in ~20 min.", self.msg("confirmed")["text"])
        self.assertNotIn("Ready in", self.msg("confirmed", prep=None)["text"])
        self.assertIn("📍 12, 100 Feet Rd", self.msg("ready")["text"])
        self.assertIn("Enjoy your meal", self.msg("completed", "delivery")["text"])

    def test_cancelled(self):
        m = self.msg("cancelled")
        self.assertEqual(m["template"], "order_cancelled")
        self.assertEqual(m["params"][3], "+91 98765 43210")
        self.assertIn("please contact +91 98765 43210", m["text"])

    def test_pending_sends_nothing(self):
        self.assertIsNone(self.msg("pending"))

    def test_defaults(self):
        m = r.order_status_message("preparing", "pickup", {"oid": "X"})
        self.assertEqual(m["params"][0], "there")
        self.assertEqual(m["params"][2], "our store")

    def test_sanitize_param(self):
        self.assertEqual(r.sanitize_template_param("a\nb\tc     d"), "a b c d")
        self.assertEqual(r.sanitize_template_param(""), "-")
        self.assertEqual(r.sanitize_template_param(None), "-")
        self.assertEqual(len(r.sanitize_template_param("x" * 500)), 200)


if __name__ == "__main__":
    unittest.main()
