import sys, unittest
from datetime import datetime
from unittest.mock import patch, MagicMock
sys.path.insert(0, r"C:\Users\visha_sl2xoym\Documents\ragby\backend")
import appointments as ap
from fastapi import HTTPException

# Monday 2 Nov 2026, 08:00 in the merchant's timezone
NOW = datetime(2026, 11, 2, 8, 0)
TUE = "2026-11-03"
SUN = "2026-11-08"


def day(start="09:00", end="18:00", enabled=True, windows=None):
    d = {"enabled": enabled, "start": start, "end": end}
    if windows is not None:
        d["windows"] = windows
    return d


def settings(**kw):
    s = {
        "project_id": "p1", "timezone": "Asia/Kolkata", "buffer_minutes": 0, "slot_capacity": 1,
        "advance_booking_days": 30,
        "working_hours": {k: day() for k in ("mon", "tue", "wed", "thu", "fri")} | {
            "sat": day(enabled=False), "sun": day(enabled=False)},
    }
    s.update(kw)
    return s


def slots(date=TUE, st=None, service=None, booked=(), busy=(), now=NOW):
    svc = {"duration_minutes": 30}
    svc.update(service or {})
    db = MagicMock()
    db.table.return_value.select.return_value.eq.return_value.eq.return_value.or_.return_value.execute.return_value.data = [
        {"start_time": a, "end_time": b} for a, b in booked]
    with patch.object(ap, "supabase", db), patch.object(ap, "_reminder_now", return_value=now):
        return ap.generate_slots(date, st or settings(), svc, list(busy))


class Slots(unittest.TestCase):
    def test_default_hours_back_to_back(self):
        r = slots()
        self.assertEqual(r[0], "09:00"); self.assertEqual(r[-1], "17:30"); self.assertEqual(len(r), 18)

    def test_closed_day(self):
        self.assertEqual(slots(SUN), [])

    def test_two_windows_leave_a_break(self):
        wh = settings()["working_hours"] | {"tue": day(windows=[{"start": "09:00", "end": "12:00"}, {"start": "14:00", "end": "17:00"}])}
        r = slots(st=settings(working_hours=wh))
        self.assertIn("11:30", r); self.assertNotIn("12:00", r); self.assertNotIn("13:30", r); self.assertIn("14:00", r)

    def test_date_override_closes_a_weekday(self):
        self.assertEqual(slots(st=settings(date_overrides=[{"date": TUE, "closed": True}])), [])

    def test_date_override_opens_a_closed_day(self):
        r = slots(SUN, settings(date_overrides=[{"date": SUN, "closed": False, "windows": [{"start": "10:00", "end": "12:00"}]}]))
        self.assertEqual(r, ["10:00", "10:30", "11:00", "11:30"])

    def test_interval_shorter_than_duration(self):
        r = slots(st=settings(slot_interval_minutes=15))
        self.assertEqual(r[:3], ["09:00", "09:15", "09:30"]); self.assertEqual(r[-1], "17:30")

    def test_buffer_after_spaces_slots(self):
        r = slots(st=settings(buffer_minutes=15))
        self.assertEqual(r[:3], ["09:00", "09:45", "10:30"])

    def test_buffer_blocks_next_to_a_booking(self):
        r = slots(st=settings(slot_interval_minutes=15, buffer_minutes=15), booked=[("10:00", "10:30")])
        self.assertNotIn("09:30", r)   # would run into the 15 min after-buffer
        self.assertIn("09:15", r)
        self.assertNotIn("10:00", r)
        self.assertIn("10:30", r)

    def test_min_notice(self):
        r = slots(st=settings(min_notice_minutes=24 * 60 + 120), now=NOW)  # tue 10:00 is the earliest cut-off
        self.assertEqual(r[0], "10:30")

    def test_past_slots_today_are_hidden(self):
        r = slots("2026-11-02", now=datetime(2026, 11, 2, 12, 10))
        self.assertEqual(r[0], "12:30")

    def test_max_per_day(self):
        self.assertEqual(slots(st=settings(max_per_day=2), booked=[("09:00", "09:30"), ("10:00", "10:30")]), [])
        self.assertTrue(slots(st=settings(max_per_day=3), booked=[("09:00", "09:30"), ("10:00", "10:30")]))

    def test_advance_booking_days(self):
        self.assertEqual(slots("2026-12-30", settings(advance_booking_days=30)), [])

    def test_capacity_label_and_limit(self):
        r = slots(st=settings(slot_capacity=2), booked=[("09:00", "09:30")])
        self.assertEqual(r[0], "09:00 (1 left)"); self.assertEqual(r[1], "09:30 (2 left)")
        r = slots(st=settings(slot_capacity=2), booked=[("09:00", "09:30"), ("09:00", "09:30")])
        self.assertNotIn("09:00", [x.split(" ")[0] for x in r])

    def test_service_capacity_beats_project(self):
        r = slots(st=settings(slot_capacity=1), service={"slot_capacity": 5})
        self.assertEqual(r[0], "09:00 (5 left)")

    def test_fixed_times_only(self):
        r = slots(service={"fixed_times": ["09:00", "13:30", "19:00"]})
        self.assertEqual(r, ["09:00", "13:30"])

    def test_service_overrides_interval_and_notice(self):
        r = slots(service={"slot_interval_minutes": 60, "min_notice_minutes": 0})
        self.assertEqual(r[:2], ["09:00", "10:00"])

    def test_google_busy_blocks_a_slot(self):
        # 03:30-04:00 UTC is 09:00-09:30 IST
        r = slots(busy=[{"start": "2026-11-03T03:30:00Z", "end": "2026-11-03T04:00:00Z"}])
        self.assertNotIn("09:00", r); self.assertIn("09:30", r)

    def test_old_working_hours_shape_still_works(self):
        self.assertEqual(ap.day_windows({"enabled": True, "start": "09:00", "end": "10:00"}), [(540, 600)])


class Validation(unittest.TestCase):
    def test_overlapping_windows_rejected(self):
        with self.assertRaises(HTTPException):
            ap._clean_working_hours({"mon": day(windows=[{"start": "09:00", "end": "12:00"}, {"start": "11:00", "end": "14:00"}])})

    def test_end_before_start_rejected(self):
        with self.assertRaises(HTTPException):
            ap._clean_working_hours({"mon": day(windows=[{"start": "12:00", "end": "09:00"}])})

    def test_enabled_day_needs_a_range(self):
        with self.assertRaises(HTTPException):
            ap._clean_working_hours({"mon": day(windows=[])})

    def test_windows_sorted_and_span_kept(self):
        out = ap._clean_working_hours({"mon": day(windows=[{"start": "14:00", "end": "17:00"}, {"start": "09:00", "end": "12:00"}])})
        self.assertEqual(out["mon"]["start"], "09:00"); self.assertEqual(out["mon"]["end"], "17:00")
        self.assertEqual(out["mon"]["windows"][0]["start"], "09:00")

    def test_date_overrides(self):
        out = ap._clean_date_overrides([{"date": "2026-12-25", "closed": True, "note": "Christmas"},
                                        {"date": "2026-12-01", "windows": [{"start": "10:00", "end": "12:00"}]}])
        self.assertEqual([o["date"] for o in out], ["2026-12-01", "2026-12-25"])
        with self.assertRaises(HTTPException):
            ap._clean_date_overrides([{"date": "2026-02-30", "closed": True}])
        with self.assertRaises(HTTPException):
            ap._clean_date_overrides([{"date": "2026-12-01"}])   # open but no hours

    def test_fixed_times(self):
        self.assertEqual(ap._clean_fixed_times(["13:30", "09:00", "09:00"]), ["09:00", "13:30"])
        self.assertIsNone(ap._clean_fixed_times([]))
        with self.assertRaises(HTTPException):
            ap._clean_fixed_times(["9am"])


class Schedules(unittest.TestCase):
    SCHEDS = [
        {"id": "s1", "is_default": True, "working_hours": {"mon": day()}, "date_overrides": []},
        {"id": "s2", "is_default": False, "working_hours": {"tue": day("18:00", "20:00")}, "date_overrides": [{"date": "2026-12-25", "closed": True}]},
    ]

    def test_service_schedule_wins_then_default_then_legacy(self):
        st = {"working_hours": {"x": 1}, "date_overrides": []}
        self.assertEqual(ap.pick_schedule(self.SCHEDS, {"schedule_id": "s2"}, st)[0], self.SCHEDS[1]["working_hours"])
        self.assertEqual(ap.pick_schedule(self.SCHEDS, {}, st)[0], self.SCHEDS[0]["working_hours"])
        self.assertEqual(ap.pick_schedule(self.SCHEDS, {"schedule_id": "gone"}, st)[0], self.SCHEDS[0]["working_hours"])
        self.assertEqual(ap.pick_schedule([], {}, st)[0], {"x": 1})

    def test_rolling_window(self):
        self.assertEqual(slots("2026-11-12", service={"booking_window_type": "rolling", "booking_window_days": 7}), [])
        self.assertTrue(slots("2026-11-05", service={"booking_window_type": "rolling", "booking_window_days": 7}))

    def test_range_window(self):
        svc = {"booking_window_type": "range", "booking_window_start": "2026-11-10", "booking_window_end": "2026-11-12"}
        self.assertEqual(slots("2026-11-09", service=svc), [])
        self.assertTrue(slots("2026-11-10", service=svc))
        self.assertEqual(slots("2026-11-13", service=svc), [])

    def test_indefinite_window_ignores_project_limit(self):
        self.assertTrue(slots("2027-03-02", settings(advance_booking_days=7), service={"booking_window_type": "indefinite"}))

    def test_service_daily_limit_counts_only_that_service(self):
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.eq.return_value.or_.return_value.execute.return_value.data = [
            {"start_time": "09:00", "end_time": "09:30", "service_id": "a"},
            {"start_time": "10:00", "end_time": "10:30", "service_id": "b"}]
        with patch.object(ap, "supabase", db), patch.object(ap, "_reminder_now", return_value=NOW):
            self.assertEqual(ap.generate_slots(TUE, settings(), {"id": "a", "duration_minutes": 30, "max_per_day": 1}, []), [])
            self.assertTrue(ap.generate_slots(TUE, settings(), {"id": "b", "duration_minutes": 30, "max_per_day": 2}, []))


class Questions(unittest.TestCase):
    Q = [{"id": "reason", "label": "Reason", "type": "text", "options": [], "required": True},
         {"id": "kind", "label": "Visit", "type": "select", "options": ["First", "Repeat"], "required": False}]

    def test_clean_questions(self):
        out = ap._clean_questions([{"label": " Why? ", "type": "select", "options": ["A", " ", "B"], "required": 1}])
        self.assertEqual(out[0]["options"], ["A", "B"]); self.assertEqual(out[0]["id"], "q1"); self.assertTrue(out[0]["required"])
        with self.assertRaises(HTTPException):
            ap._clean_questions([{"label": "", "type": "text"}])
        with self.assertRaises(HTTPException):
            ap._clean_questions([{"label": "x", "type": "select", "options": []}])
        with self.assertRaises(HTTPException):
            ap._clean_questions([{"label": "x", "type": "text"}] * 11)

    def test_answers_required_and_options(self):
        with self.assertRaises(ap.AnswerError):
            ap.clean_answers(self.Q, {})
        with self.assertRaises(ap.AnswerError):
            ap.clean_answers(self.Q, {"reason": "pain", "kind": "Other"})
        self.assertEqual(ap.clean_answers(self.Q, {"reason": "pain"}), [{"question": "Reason", "answer": "pain"}])

    def test_booking_window_validation(self):
        with self.assertRaises(HTTPException):
            ap._check_booking_window("rolling", None, None, None)
        with self.assertRaises(HTTPException):
            ap._check_booking_window("range", None, "2026-12-02", "2026-12-01")
        ap._check_booking_window("range", None, "2026-12-01", "2026-12-02")


class Rules(unittest.TestCase):
    APPT = {"appointment_date": "2026-11-03", "start_time": "12:00:00"}

    def problem(self, settings=None, service=None, now=NOW):
        with patch.object(ap, "_reminder_now", return_value=now):
            return ap.change_cutoff_problem(self.APPT, settings or {}, service)

    def test_no_cutoff_means_always_allowed(self):
        self.assertIsNone(self.problem())
        self.assertIsNone(self.problem({"cancel_cutoff_hours": 0}))

    def test_cutoff_blocks_late_changes(self):
        # now is Mon 08:00; the appointment is Tue 12:00 = 28 hours away
        self.assertIsNone(self.problem({"cancel_cutoff_hours": 24}))
        self.assertIn("48 hours", self.problem({"cancel_cutoff_hours": 48}))
        self.assertIn("1 hour ", self.problem({"cancel_cutoff_hours": 1}, now=datetime(2026, 11, 3, 11, 30)) + " ")

    def test_service_cutoff_beats_project(self):
        self.assertIsNone(self.problem({"cancel_cutoff_hours": 48}, {"cancel_cutoff_hours": 2}))

    def test_booking_details_line(self):
        self.assertEqual(ap.booking_details({"location": "Main St clinic", "confirmation_note": "Bring reports"}),
                         "Where: Main St clinic. Bring reports")
        self.assertEqual(ap.booking_details({}), "")

    def test_approval_only_with_free_services(self):
        ap._check_approval(True, "free"); ap._check_approval(False, "hold_to_confirm"); ap._check_approval(None, None)
        with self.assertRaises(HTTPException):
            ap._check_approval(True, "hold_to_confirm")

    def test_pending_requests_hold_their_slot(self):
        db = MagicMock()
        chain = db.table.return_value.select.return_value.eq.return_value.eq.return_value.or_
        chain.return_value.execute.return_value.data = []
        with patch.object(ap, "supabase", db), patch.object(ap, "_reminder_now", return_value=NOW):
            ap.generate_slots(TUE, settings(), {"duration_minutes": 30}, [])
        self.assertIn("pending_approval", chain.call_args.args[0])


class LinkAndLengths(unittest.TestCase):
    def test_slug_rules(self):
        self.assertEqual(ap.clean_slug(" Clinic "), "clinic")
        self.assertIsNone(ap.clean_slug(""))
        self.assertIsNone(ap.clean_slug(None))
        for bad in ("ab", "has space", "UPPER_case", "-start", "api", "slots", "a" * 41,
                    "121be061-ff70-40d2-a99e-3d83bad09137"):
            with self.assertRaises(HTTPException, msg=bad):
                ap.clean_slug(bad)

    def test_duration_options(self):
        self.assertEqual(ap._clean_duration_options([60, 15, 30, 30]), [15, 30, 60])
        self.assertIsNone(ap._clean_duration_options([]))
        self.assertIsNone(ap._clean_duration_options(None))
        with self.assertRaises(HTTPException):
            ap._clean_duration_options([3])
        self.assertEqual(ap.allowed_durations({"duration_minutes": 30, "duration_options": [15, 60]}), [15, 30, 60])

    def test_slots_for_a_chosen_length(self):
        svc = {"id": "s", "name": "Consult", "is_active": True, "duration_minutes": 30, "duration_options": [60]}
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = settings()
        with patch.object(ap, "supabase", db), patch.object(ap, "get_service", return_value=svc), \
             patch.object(ap, "_load_schedules", return_value=[]), \
             patch.object(ap, "generate_slots", return_value=[]) as gen:
            ap.get_available_slots("p1", TUE, "s", 60)
            self.assertEqual(gen.call_args.args[2]["duration_minutes"], 60)
            with self.assertRaises(ValueError):
                ap.get_available_slots("p1", TUE, "s", 45)

    def test_longer_slots_fit_fewer_times(self):
        self.assertLess(len(slots(service={"duration_minutes": 60})), len(slots(service={"duration_minutes": 30})))


if __name__ == "__main__":
    unittest.main()
