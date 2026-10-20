import unittest
from datetime import datetime, timezone
from unittest import mock

import shop
import shop_invoice as inv
from tests.fake_supabase import FakeSupabase

PID = "11111111-1111-1111-1111-111111111111"
OID = "22222222-2222-2222-2222-222222222222"


class Numbering(unittest.TestCase):
    def test_format(self):
        self.assertEqual(inv.format_invoice_number("2026-27", 1), "INV/2026-27/0001")
        self.assertEqual(inv.format_invoice_number("2026-27", 12345), "INV/2026-27/12345")

    def test_fy_boundary_is_in_india_time(self):
        # 31 Mar 2027 18:29 UTC = 23:59 IST: still FY 2026-27.
        self.assertEqual(inv.financial_year(datetime(2027, 3, 31, 18, 29, tzinfo=timezone.utc)), "2026-27")
        # 31 Mar 2027 18:30 UTC = 1 Apr 00:00 IST: FY 2027-28.
        self.assertEqual(inv.financial_year(datetime(2027, 3, 31, 18, 30, tzinfo=timezone.utc)), "2027-28")
        self.assertEqual(inv.financial_year(datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)), "2026-27")
        self.assertEqual(inv.financial_year(datetime(2026, 1, 15, tzinfo=timezone.utc)), "2025-26")
        self.assertEqual(inv.financial_year(datetime(2099, 6, 1, tzinfo=timezone.utc)), "2099-00")


class Words(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual(inv.amount_in_words(0), "Rupees Zero Only")
        self.assertEqual(inv.amount_in_words(1), "Rupees One Only")
        self.assertEqual(inv.amount_in_words(1234.5),
                         "Rupees One Thousand Two Hundred Thirty Four and Fifty Paise Only")
        self.assertEqual(inv.amount_in_words(100000), "Rupees One Lakh Only")
        self.assertEqual(inv.amount_in_words(12345678.09),
                         "Rupees One Crore Twenty Three Lakh Forty Five Thousand Six Hundred Seventy Eight"
                         " and Nine Paise Only")
        self.assertEqual(inv.amount_in_words(119.99), "Rupees One Hundred Nineteen and Ninety Nine Paise Only")


class Split(unittest.TestCase):
    def test_halves_add_up_exactly(self):
        for paise in range(0, 2000):
            tax = paise / 100
            c, s = inv.split_tax(tax)
            self.assertEqual(round((c + s) * 100), paise)
            self.assertLessEqual(abs(c - s), 0.0100001)

    def test_odd_paisa(self):
        self.assertEqual(inv.split_tax(0.05), (0.02, 0.03))
        self.assertEqual(inv.split_tax(18), (9.0, 9.0))

    def test_invoice_totals_add_up(self):
        order = {"id": OID, "project_id": PID, "status": "confirmed", "payment_status": "paid",
                 "items": [{"name": "A", "price": 10.5, "quantity": 3, "gst_percent": 5},
                           {"name": "B", "price": 99, "quantity": 1, "gst_percent": 18}],
                 "items_discount": 10, "subtotal": 130.5,
                 "gst_breakdown": [{"rate": 5, "taxable": 29.09, "tax": 1.45},
                                   {"rate": 18, "taxable": 91.41, "tax": 16.45}],
                 "gst_amount": 17.9, "total": 138.4, "invoice_number": "INV/2026-27/0001",
                 "seller_snapshot": {"gstin": "29ABCDE1234F1Z5", "state": "Karnataka"}}
        out = inv.build_invoice(order)
        self.assertEqual(out["title"], "Tax Invoice")
        self.assertAlmostEqual(out["cgst_total"] + out["sgst_total"], out["tax_total"], places=6)
        self.assertAlmostEqual(sum(r["cgst"] + r["sgst"] for r in out["tax_rows"]), 17.9, places=6)
        self.assertAlmostEqual(sum(l["taxable"] for l in out["items"]), 120.5, places=6)

    def test_old_order_falls_back_to_gst_amount_and_receipt_without_gstin(self):
        order = {"id": OID, "project_id": PID, "status": "confirmed", "items": [], "gst_amount": 9.01,
                 "total": 109.01, "phone_number": "919876543210", "seller_snapshot": {}}
        out = inv.build_invoice(order)
        self.assertEqual(out["title"], "Receipt")
        self.assertEqual(len(out["tax_rows"]), 1)
        self.assertEqual(out["tax_rows"][0]["cgst"] + out["tax_rows"][0]["sgst"], 9.01)
        self.assertNotIn("9876543210", out["customer"]["phone"])
        self.assertEqual(inv.build_invoice(order, public=False)["customer"]["phone"], "+919876543210")


class Gstin(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertEqual(inv.normalize_gstin(" 29abcde1234f1z5 "), "29ABCDE1234F1Z5")
        self.assertIsNone(inv.normalize_gstin(""))
        for bad in ("29ABCDE1234F1Z", "29ABCDE1234F1X5", "ABCDE1234F1Z529", "29ABCDE1234F0Z5"):
            with self.assertRaises(ValueError):
                inv.normalize_gstin(bad)


def _fake_issue_rpc(db):
    """Python stand-in for shop_issue_invoice: same guards and numbering."""
    counters = {}

    def fn(params):
        order = next((o for o in db.tables["orders"] if o["id"] == params["p_order_id"]), None)
        if not order:
            return []
        if order.get("invoice_number"):
            return [{"out_invoice_number": order["invoice_number"], "out_invoice_date": order["invoice_date"]}]
        if order.get("status") == "cancelled":
            return []
        fy = inv.financial_year(datetime.now(timezone.utc))
        key = (order["project_id"], fy)
        counters[key] = counters.get(key, 0) + 1
        order["invoice_number"] = inv.format_invoice_number(fy, counters[key])
        order["invoice_date"] = "2026-10-20T00:00:00+00:00"
        order["seller_snapshot"] = params["p_snapshot"]
        return [{"out_invoice_number": order["invoice_number"], "out_invoice_date": order["invoice_date"]}]
    return fn


class Issue(unittest.TestCase):
    def setUp(self):
        self.db = FakeSupabase(tables={
            "orders": [
                {"id": OID, "project_id": PID, "status": "confirmed", "payment_status": "paid", "branch_id": None},
                {"id": "33333333-3333-3333-3333-333333333333", "project_id": PID, "status": "pending",
                 "payment_status": "unpaid"},
                {"id": "44444444-4444-4444-4444-444444444444", "project_id": PID, "status": "confirmed",
                 "payment_status": "unpaid", "payment_method": "offline"},
            ],
            "shop_config": [{"project_id": PID, "store_name": "Cafe", "gstin": "29ABCDE1234F1Z5",
                             "state": "Karnataka"}],
        })
        self.db.rpcs["shop_issue_invoice"] = _fake_issue_rpc(self.db)
        p = mock.patch.object(shop, "supabase", self.db)
        p.start()
        self.addCleanup(p.stop)

    def test_idempotent(self):
        order = dict(self.db.tables["orders"][0])
        first = inv.issue_invoice(order)
        self.assertTrue(first["invoice_number"].endswith("/0001"))
        self.assertEqual(first["seller_snapshot"]["gstin"], "29ABCDE1234F1Z5")
        # A stale copy without the number: the DB returns the same number.
        again = inv.issue_invoice(dict(order))
        self.assertEqual(again["invoice_number"], first["invoice_number"])
        # Edits after issue don't change the frozen snapshot.
        self.db.tables["shop_config"][0]["gstin"] = None
        self.assertEqual(inv.issue_invoice(dict(order))["seller_snapshot"]["gstin"], "29ABCDE1234F1Z5")
        self.assertEqual(sum(1 for c in self.db.rpc_calls if c[0] == "shop_issue_invoice"), 3)

    def test_sequential_and_skips_pending(self):
        self.assertIsNone(inv.issue_invoice(dict(self.db.tables["orders"][1])))
        a = inv.issue_invoice(dict(self.db.tables["orders"][0]))
        b = inv.issue_invoice(dict(self.db.tables["orders"][2]))
        self.assertTrue(a["invoice_number"].endswith("/0001"))
        self.assertTrue(b["invoice_number"].endswith("/0002"))

    def test_failure_never_raises(self):
        self.db.rpcs.pop("shop_issue_invoice")
        self.assertIsNone(inv.issue_invoice(dict(self.db.tables["orders"][0])))
        self.assertEqual(shop.with_invoice({"id": OID, "project_id": PID, "status": "confirmed"}),
                         {"id": OID, "project_id": PID, "status": "confirmed"})


if __name__ == "__main__":
    unittest.main()
