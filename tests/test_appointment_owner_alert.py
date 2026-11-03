import sys, unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0, r"C:\Users\visha_sl2xoym\Documents\ragby\backend")
import appointments as ap
import template_library

APPT = {"id": "abcdef12-0000-0000-0000-000000000000", "project_id": "p1", "customer_phone": "919999999999",
        "customer_name": "Asha", "service_name": "Dental checkup", "appointment_date": "2026-11-03",
        "start_time": "12:00:00", "notes": "first visit"}


def fake_db(settings, wa={"phone_number_id": "pid", "access_token": "tok"}):
    def table(name):
        t = MagicMock()
        data = settings if name == "appointment_settings" else wa
        t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = data
        return t
    db = MagicMock()
    db.table.side_effect = table
    return db


class OwnerAlert(unittest.TestCase):
    def run_alert(self, settings, window=True, tpl_ok=True, wa={"phone_number_id": "pid", "access_token": "tok"}, **kw):
        with patch.object(ap, "supabase", fake_db(settings, wa)), \
             patch("shop._owner_window_open", return_value=window), \
             patch.object(ap, "_send_appt_template", return_value=tpl_ok) as tpl, \
             patch("whatsapp.send_whatsapp_message") as txt:
            ap.notify_owner_booking(APPT, **kw)
            return tpl, txt

    ON = {"notify_owner_booking": True, "owner_phone": "919888888888"}

    def test_window_open_sends_text_only(self):
        tpl, txt = self.run_alert(self.ON, window=True, answers=[{"question": "Reason", "answer": "pain"}])
        tpl.assert_not_called(); txt.assert_called_once()
        body = txt.call_args.kwargs["text"]
        self.assertIn("Dental checkup", body); self.assertIn("Reason: pain", body); self.assertIn("first visit", body)

    def test_window_closed_sends_template(self):
        tpl, txt = self.run_alert(self.ON, window=False)
        tpl.assert_called_once(); txt.assert_not_called()
        self.assertEqual(tpl.call_args.args[1], "919888888888")
        self.assertEqual(tpl.call_args.args[2][0], "New booking")

    def test_template_missing_falls_back_to_text(self):
        tpl, txt = self.run_alert(self.ON, window=False, tpl_ok=False)
        tpl.assert_called_once(); txt.assert_called_once()

    def test_switch_off_sends_nothing(self):
        tpl, txt = self.run_alert({**self.ON, "notify_owner_booking": False})
        tpl.assert_not_called(); txt.assert_not_called()

    def test_no_phone_sends_nothing(self):
        tpl, txt = self.run_alert({"notify_owner_booking": True, "owner_phone": None})
        tpl.assert_not_called(); txt.assert_not_called()

    def test_no_whatsapp_connected_sends_nothing(self):
        tpl, txt = self.run_alert(self.ON, wa=None)
        tpl.assert_not_called(); txt.assert_not_called()

    def test_rescheduled_label(self):
        tpl, _ = self.run_alert(self.ON, window=False, event="rescheduled")
        self.assertEqual(tpl.call_args.args[2][0], "Rescheduled booking")

    def test_a_failure_never_raises(self):
        with patch.object(ap, "supabase", MagicMock(side_effect=RuntimeError("db down"))):
            ap.notify_owner_booking(APPT)

    def test_library_template_follows_meta_rules(self):
        t = next(x for x in template_library.TEMPLATE_LIBRARY if x["name"] == ap.OWNER_BOOKING_TEMPLATE)
        body = t["components"][0]["text"]
        self.assertFalse(body.startswith("{{")); self.assertFalse(body.endswith("}}"))
        self.assertEqual(body.count("{{"), len(t["example_body"]))


class Cancel(unittest.TestCase):
    def run_cancel(self, notify_customer):
        row = {**APPT, "status": "confirmed", "answers": None}
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = row
        db.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = row
        with patch.object(ap, "supabase", db), patch.object(ap, "notify_owner_booking") as alert,              patch.object(ap, "_send_appt_template", return_value=True):
            ap.cancel_appointment(APPT["id"], notify_customer=notify_customer)
            return alert

    def test_customer_cancel_alerts_owner(self):
        alert = self.run_cancel(notify_customer=False)
        alert.assert_called_once()
        self.assertEqual(alert.call_args.args[1], "cancelled")

    def test_dashboard_cancel_does_not_alert_owner(self):
        self.run_cancel(notify_customer=True).assert_not_called()


class Reminders(unittest.TestCase):
    def test_offsets_default_to_reminder_hours(self):
        self.assertEqual(ap.reminder_offsets_for({}), [24])
        self.assertEqual(ap.reminder_offsets_for({"reminder_hours": 6}), [6])
        self.assertEqual(ap.reminder_offsets_for({"reminder_offsets": [], "reminder_hours": 12}), [12])

    def test_offsets_sorted_deduped_and_capped(self):
        self.assertEqual(ap.reminder_offsets_for({"reminder_offsets": [2, 24, 24, 48, 6, 1, 500, 0]}), [48, 24, 6, 2])

    def test_stage_walks_through_the_offsets(self):
        offs = [24, 2]
        self.assertEqual(ap.reminder_stage_now(offs, 30, 0), 0)    # too early
        self.assertEqual(ap.reminder_stage_now(offs, 23, 0), 1)    # first reminder due
        self.assertEqual(ap.reminder_stage_now(offs, 20, 1), 1)    # already sent
        self.assertEqual(ap.reminder_stage_now(offs, 1.5, 1), 2)   # second due
        self.assertEqual(ap.reminder_stage_now(offs, 1.5, 2), 2)   # all done

    def test_late_booking_sends_one_message_for_all_due(self):
        self.assertEqual(ap.reminder_stage_now([24, 2], 1, 0), 2)

    def test_nothing_after_it_started(self):
        self.assertEqual(ap.reminder_stage_now([24, 2], -1, 0), 0)


if __name__ == "__main__":
    unittest.main()
