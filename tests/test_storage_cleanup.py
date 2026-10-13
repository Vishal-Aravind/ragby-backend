"""Storage clean-up decisions and reference finding. No network.

Run from backend/:  python -m unittest tests.test_storage_cleanup -v
"""
import json
import unittest
from datetime import datetime, timedelta, timezone

import storage_cleanup as sc

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=30)


class PlanTests(unittest.TestCase):
    def test_new_unused_file_is_only_noted(self):
        note, unnote, delete = sc.plan_cleanup([("u/message_media/a.jpg", OLD)], set(), {}, NOW)
        self.assertEqual((note, unnote, delete), (["u/message_media/a.jpg"], [], []))

    def test_deleted_only_after_seven_days_unused(self):
        files = [("u/message_media/a.jpg", OLD), ("u/message_media/b.jpg", OLD)]
        noted = {"u/message_media/a.jpg": NOW - timedelta(days=8), "u/message_media/b.jpg": NOW - timedelta(days=3)}
        _, _, delete = sc.plan_cleanup(files, set(), noted, NOW)
        self.assertEqual(delete, ["u/message_media/a.jpg"])

    def test_used_again_is_unnoted_and_kept(self):
        noted = {"u/message_media/a.jpg": NOW - timedelta(days=20)}
        note, unnote, delete = sc.plan_cleanup([("u/message_media/a.jpg", OLD)], {"u/message_media/a.jpg"}, noted, NOW)
        self.assertEqual((note, unnote, delete), ([], ["u/message_media/a.jpg"], []))

    def test_files_in_use_or_fresh_are_never_touched(self):
        files = [("u/products/used.jpg", OLD), ("u/products/fresh.jpg", NOW - timedelta(hours=2))]
        note, _, delete = sc.plan_cleanup(files, {"u/products/used.jpg"}, {}, NOW)
        self.assertEqual((note, delete), ([], []))

    def test_rows_for_files_already_gone_are_cleared(self):
        _, unnote, _ = sc.plan_cleanup([], set(), {"u/products/gone.jpg": OLD}, NOW)
        self.assertEqual(unnote, ["u/products/gone.jpg"])

    def test_delete_is_capped_per_run(self):
        files = [(f"u/web-image/{i}.jpg", OLD) for i in range(sc.MAX_DELETE_PER_RUN + 50)]
        noted = {p: NOW - timedelta(days=10) for p, _ in files}
        _, _, delete = sc.plan_cleanup(files, set(), noted, NOW)
        self.assertEqual(len(delete), sc.MAX_DELETE_PER_RUN)


class FolderTests(unittest.TestCase):
    def test_only_known_upload_folders_are_cleaned(self):
        files = [("u/message_media/a.jpg", OLD), ("u/new-feature/b.jpg", OLD), ("loose.jpg", OLD)]
        noted = {p: NOW - timedelta(days=30) for p, _ in files}
        note, unnote, delete = sc.plan_cleanup(files, set(), noted, NOW)
        self.assertEqual(delete, ["u/message_media/a.jpg"])
        # files outside the known folders are un-noted, never deleted
        self.assertEqual(unnote, ["loose.jpg", "u/new-feature/b.jpg"])


class ReferenceTests(unittest.TestCase):
    def test_finds_urls_inside_json(self):
        base = "https://x.supabase.co/storage/v1/object/public/flow-media/"
        row = {"content": {"media_url": base + "u1/message_media/1.jpg",
                           "cards": [{"image": base + "u1/web-image/2.png?v=1"}]},
               "page_json": [{"props": {"images": [base + "u1/event-pages/3.webp"]}}]}
        found = {m.group(1) for m in sc.PATH_RE.finditer(json.dumps(row))}
        self.assertEqual(found, {"u1/message_media/1.jpg", "u1/web-image/2.png", "u1/event-pages/3.webp"})


if __name__ == "__main__":
    unittest.main()
