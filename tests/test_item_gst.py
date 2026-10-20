import unittest

import shop
import shop_branches
import shop_rules as r
from tests.test_shop_offers import ShopHarness, PID, PROD, CAT


def line(value, rate=None):
    return {"line_total": value, "gst_percent": rate}


class ComputeTotals(unittest.TestCase):
    def test_same_as_the_old_single_rate_math(self):
        lines = [line(200), line(300)]
        new = r.compute_totals(lines, 18, 40, 50, 0)
        old = r.apply_offer_totals(500, 18, 40, 50, 0)
        self.assertEqual(new["gst_amount"], old["gst_amount"])
        self.assertEqual(new["total"], old["total"])
        self.assertEqual(new["taxable_subtotal"], old["taxable_subtotal"])

    def test_item_without_a_rate_uses_the_shop_rate(self):
        t = r.compute_totals([line(100, None)], 12, 0, 0, 0)
        self.assertEqual(t["gst_amount"], 12.0)
        self.assertEqual(t["line_rates"], [12.0])

    def test_zero_means_exempt_not_shop_rate(self):
        t = r.compute_totals([line(100, 0)], 18, 0, 0, 0)
        self.assertEqual(t["gst_amount"], 0.0)
        self.assertEqual(t["total"], 100.0)

    def test_mixed_rates_group_by_rate(self):
        t = r.compute_totals([line(200, 5), line(100, 18), line(100, 5)], 18, 0, 0, 0)
        self.assertEqual(t["gst_breakdown"], [
            {"rate": 5.0, "taxable": 300.0, "tax": 15.0},
            {"rate": 18.0, "taxable": 100.0, "tax": 18.0},
        ])
        self.assertEqual(t["gst_amount"], 33.0)
        self.assertEqual(t["total"], 433.0)

    def test_discount_is_shared_in_proportion(self):
        # 100 @5% and 300 @18%, 40 off: 10 comes off the first, 30 off the second
        t = r.compute_totals([line(100, 5), line(300, 18)], 18, 0, 40, 0)
        self.assertEqual(t["taxable_subtotal"], 360.0)
        self.assertEqual(t["gst_breakdown"][0], {"rate": 5.0, "taxable": 90.0, "tax": 4.5})
        self.assertEqual(t["gst_breakdown"][1], {"rate": 18.0, "taxable": 270.0, "tax": 48.6})

    def test_discount_shares_add_up_exactly(self):
        # an awkward split must not lose or gain a paisa
        t = r.compute_totals([line(33.33, 5), line(33.33, 12), line(33.34, 18)], 18, 0, 10, 0)
        self.assertEqual(t["taxable_subtotal"], 90.0)

    def test_discount_cannot_go_below_zero(self):
        t = r.compute_totals([line(100, 18)], 18, 0, 500, 0)
        self.assertEqual(t["taxable_subtotal"], 0.0)
        self.assertEqual(t["gst_amount"], 0.0)

    def test_delivery_fee_is_not_taxed_and_delivery_discount_applies(self):
        t = r.compute_totals([line(100, 18)], 18, 40, 0, 40)
        self.assertEqual(t["gst_amount"], 18.0)
        self.assertEqual(t["total"], 118.0)

    def test_empty_cart(self):
        t = r.compute_totals([], 18, 0, 0, 0)
        self.assertEqual(t["total"], 0.0)
        self.assertEqual(t["gst_breakdown"], [])


class GstTextLines(unittest.TestCase):
    def test_single_rate_one_line(self):
        b = [{"rate": 18.0, "taxable": 100.0, "tax": 18.0}]
        self.assertEqual(r.gst_text_lines(b, 18, "₹"), ["GST (18%): ₹18.00"])

    def test_two_rates_two_lines(self):
        b = [{"rate": 5.0, "taxable": 300.0, "tax": 15.0}, {"rate": 18.0, "taxable": 100.0, "tax": 18.0}]
        self.assertEqual(r.gst_text_lines(b, 33, "₹"), ["GST 5%: ₹15.00", "GST 18%: ₹18.00"])

    def test_old_order_without_breakdown(self):
        self.assertEqual(r.gst_text_lines([], 12.5, "₹"), ["GST: ₹12.50"])
        self.assertEqual(r.gst_text_lines(None, 0, "₹"), [])

    def test_exempt_only_shows_nothing(self):
        self.assertEqual(r.gst_text_lines([{"rate": 0.0, "taxable": 100.0, "tax": 0.0}], 0, "₹"), [])


class PriceOrderPerItemGst(ShopHarness):
    """Through the real pricing path: items keep their own rate, an item with
    no rate uses the shop rate (5% in this fixture), 0 is exempt."""

    def test_mixed_rates_end_to_end(self):
        prods = self.db.tables["products"]
        prods[0].update({"price": 100.0, "gst_percent": 12})                 # Ghee Pongal at 12%
        prods.append({"id": "22222222-2222-4222-8222-222222222222", "project_id": PID, "catalog_id": CAT,
                      "name": "Tea", "price": 50.0, "is_available": True, "image_url": None,
                      "shopify_variant_id": None, "gst_percent": None})      # no rate -> shop rate 5%
        prods.append({"id": "33333333-3333-4333-8333-333333333333", "project_id": PID, "catalog_id": CAT,
                      "name": "Milk", "price": 20.0, "is_available": True, "image_url": None,
                      "shopify_variant_id": None, "gst_percent": 0})         # exempt
        branch = self.db.tables["shop_branches"][0]
        items = [shop.CartItem(product_id=PROD, quantity=1),
                 shop.CartItem(product_id="22222222-2222-4222-8222-222222222222", quantity=2),
                 shop.CartItem(product_id="33333333-3333-4333-8333-333333333333", quantity=1)]
        p = shop_branches.price_order(PID, branch, items, "pickup")
        self.assertEqual(p["subtotal"], 220.0)
        self.assertEqual({b["rate"]: b["tax"] for b in p["gst_breakdown"]}, {0.0: 0.0, 5.0: 5.0, 12.0: 12.0})
        self.assertEqual(p["gst_amount"], 17.0)
        self.assertEqual(p["total"], 237.0)
        # the rate actually charged is saved with each item
        self.assertEqual([i["gst_percent"] for i in p["items_data"]], [12.0, 5.0, 0.0])


if __name__ == "__main__":
    unittest.main()
