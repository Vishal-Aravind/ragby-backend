"""PUT /shop-config: an agent (no min_role) may still only flip
accepting_orders (the day-to-day pause/resume) and the always-agent-
editable fields; any other store field (location, hours, delivery,
min order...) now requires admin, matching the min_role gate those
fields had on shop_branches before single-store removed that table.

Run from backend/: python -m unittest tests.test_shop_config_auth -v
"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import shop

PID = "11111111-1111-1111-1111-111111111111"
UID = "22222222-2222-2222-2222-222222222222"


class UpdateShopConfigAuthTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_require(user_id, project_id, tab=None, min_role=None):
            self.calls.append({"tab": tab, "min_role": min_role})

        db = mock.MagicMock()
        db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = {"id": "x"}
        db.table.return_value.update.return_value.eq.return_value.execute.return_value = None
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {"project_id": PID}

        patches = [
            mock.patch.object(shop, "require_project_access", side_effect=fake_require),
            mock.patch.object(shop, "supabase", db),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def call(self, **fields):
        body = shop.ShopConfigUpdate(**fields)
        asyncio.run(shop.update_shop_config(PID, body, user=SimpleNamespace(id=UID)))
        return self.calls[-1]

    def test_accepting_orders_alone_needs_no_admin(self):
        self.assertIsNone(self.call(accepting_orders=True)["min_role"])

    def test_always_agent_fields_need_no_admin(self):
        self.assertIsNone(self.call(store_name="Annapoorna")["min_role"])
        self.assertIsNone(self.call(is_enabled=True)["min_role"])
        self.assertIsNone(self.call(notify_owner_whatsapp=True)["min_role"])

    def test_a_moved_store_field_needs_admin(self):
        for kwargs in (
            {"hours": {"mon": {"closed": False, "open": "09:00", "close": "18:00"}}},
            {"delivery_per_km": 10},
            {"min_order_value": 100},
            {"address": "12 Main St"},
            {"lat": 12.9, "lng": 80.2},
        ):
            self.assertEqual(self.call(**kwargs)["min_role"], "admin", kwargs)

    def test_pause_plus_a_store_field_together_still_needs_admin(self):
        self.assertEqual(self.call(accepting_orders=False, prep_minutes=20)["min_role"], "admin")

    def test_pause_plus_an_always_agent_field_needs_no_admin(self):
        self.assertIsNone(self.call(accepting_orders=False, store_name="Annapoorna")["min_role"])


if __name__ == "__main__":
    unittest.main()
