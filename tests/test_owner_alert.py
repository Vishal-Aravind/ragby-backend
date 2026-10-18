import sys, unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0, r"C:\Users\visha_sl2xoym\Documents\ragby\backend")
import shop

ORDER = {"id": "abcdef12-0000-0000-0000-000000000000", "project_id": "p1", "phone_number": "919999999999",
         "customer_name": "Asha", "items": [{"name": "Tea", "quantity": 1, "price": 10}], "subtotal": 10,
         "total": 10, "payment_status": "unpaid", "fulfilment": "pickup"}
BRANCH = {"name": "Main", "phone": "919888888888"}
CFG = {"currency": "₹"}


class T(unittest.TestCase):
    def run_alert(self, window, tpl_ok=True, cfg=CFG, branch=BRANCH, **kw):
        with patch.object(shop, "_owner_window_open", return_value=window), \
             patch.object(shop, "_send_order_template", return_value=tpl_ok) as tpl, \
             patch("whatsapp.send_whatsapp_message") as txt:
            shop.send_owner_order_alert(ORDER, branch, cfg, "pid", "tok", **kw)
            return tpl, txt

    def test_window_open_sends_text_only(self):
        tpl, txt = self.run_alert(True)
        tpl.assert_not_called(); txt.assert_called_once()

    def test_window_closed_sends_template_only(self):
        tpl, txt = self.run_alert(False)
        tpl.assert_called_once(); txt.assert_not_called()
        params = tpl.call_args[0][2]
        self.assertEqual(params[0], "ABCDEF12"); self.assertEqual(params[4], "pickup")

    def test_template_missing_falls_back_to_text(self):
        tpl, txt = self.run_alert(False, tpl_ok=False)
        tpl.assert_called_once(); txt.assert_called_once()

    def test_branch_switch_off_sends_nothing(self):
        tpl, txt = self.run_alert(False, branch={**BRANCH, "notify_owner_whatsapp": False})
        tpl.assert_not_called(); txt.assert_not_called()

    def test_branch_switch_missing_means_on(self):
        tpl, txt = self.run_alert(True)
        txt.assert_called_once()

    def test_branch_without_phone_sends_nothing(self):
        tpl, txt = self.run_alert(False, branch={"name": "Main"}, cfg={"currency": "₹", "store_phone": "919000000000"})
        tpl.assert_not_called(); txt.assert_not_called()

    def test_goes_to_the_branch_phone(self):
        tpl, _ = self.run_alert(False)
        self.assertEqual(tpl.call_args[0][0], "919888888888")

    def test_refund_note_goes_in_payment_param(self):
        tpl, _ = self.run_alert(False, payment_note="paid after the order was cancelled - refund or fulfil")
        self.assertIn("refund", tpl.call_args[0][2][5])


if __name__ == "__main__":
    unittest.main()
