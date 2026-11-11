"""Single-store shop: store settings validation, the store view built from
shop_config, and setup warnings. Run from backend/:
python -m unittest tests.test_shop_store -v
"""
import unittest
from unittest import mock

from fastapi import HTTPException

import shop_store


class StoreFieldsTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch("phones.default_region_for_project", lambda pid: "IN")
        p.start()
        self.addCleanup(p.stop)

    def fields(self, **data):
        return shop_store.store_fields(data, "p1")

    def test_only_sent_keys_come_back(self):
        self.assertEqual(self.fields(pickup_enabled=False), {"pickup_enabled": False})
        self.assertEqual(self.fields(), {})

    def test_null_clears(self):
        out = self.fields(hours=None, free_delivery_above=None, max_delivery_km=None, prep_minutes=None,
                          lat=None, lng=None)
        self.assertEqual(out, {"hours": None, "free_delivery_above": None, "max_delivery_km": None,
                               "prep_minutes": None, "lat": None, "lng": None})

    def test_empty_money_means_zero(self):
        out = self.fields(delivery_per_km=None, delivery_base_fee=None, min_order_value=None)
        self.assertEqual(out, {"delivery_per_km": 0, "delivery_base_fee": 0, "min_order_value": 0})

    def test_rejects_bad_values(self):
        for bad in ({"lat": 12.9}, {"max_delivery_km": 500}, {"prep_minutes": 20.5},
                    {"pincodes": ["12"]}, {"timezone": "Nope/Nowhere"}, {"pickup_enabled": "yes"}):
            with self.assertRaises(HTTPException, msg=bad):
                self.fields(**bad)

    def test_pincodes_deduped(self):
        self.assertEqual(self.fields(pincodes=["600042", 600042, " 600041 "])["pincodes"], ["600042", "600041"])


class StoreViewTests(unittest.TestCase):
    def test_defaults_for_a_new_shop(self):
        s = shop_store.store_view({"store_name": "  Annapoorna "})
        self.assertEqual(s["name"], "Annapoorna")
        self.assertTrue(s["pickup_enabled"])
        self.assertFalse(s["delivery_enabled"])
        self.assertTrue(s["accepting_orders"])
        self.assertEqual(s["timezone"], "Asia/Kolkata")
        self.assertIsNone(s["phone"])

    def test_phone_from_settings(self):
        with mock.patch("phones.default_region_for_project", lambda pid: "IN"):
            s = shop_store.store_view({"project_id": "p1", "store_phone": "+91 98765 43210"})
        self.assertEqual(s["phone"], "919876543210")

    def test_pause_and_shop_off(self):
        self.assertEqual(shop_store.status_for(shop_store.store_view({"accepting_orders": False}), {})["reason"], "paused")
        self.assertEqual(shop_store.status_for(shop_store.store_view({}), {"is_enabled": False})["reason"], "inactive")

    def test_warnings(self):
        w = shop_store.store_warnings(shop_store.store_view({"delivery_enabled": True}))
        self.assertIn("delivery_not_configured", w)
        self.assertIn("no_phone", w)
        # pincode-only delivery needs no map pin
        ok = shop_store.store_view({"delivery_enabled": True, "pincodes": ["600042"], "store_phone": "919876543210"})
        self.assertEqual(shop_store.store_warnings(ok), [])
        # a distance needs the pin
        far = shop_store.store_view({"delivery_enabled": True, "max_delivery_km": 5, "store_phone": "919876543210"})
        self.assertEqual(shop_store.store_warnings(far), ["delivery_not_configured"])


if __name__ == "__main__":
    unittest.main()
