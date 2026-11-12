"""Config & Risk shows which currency each money / price field is in."""
from dashboard.app import MARKET_CURRENCY
from dashboard.schema import FIELDS


def test_money_and_price_fields_are_marked():
    assert FIELDS["risk.max_daily_loss"]["unit"] == "money"
    assert FIELDS["risk.max_loss_per_trade"]["unit"] == "money"
    for key in ("risk.stop_loss_points", "risk.take_profit_points",
                "trailing_stop.trigger_points", "trailing_stop.trail_distance"):
        assert FIELDS[key]["unit"] == "price"
    # percentages and counts carry no currency
    assert "unit" not in FIELDS["risk.stop_loss_pct"] and "unit" not in FIELDS["risk.max_open_positions"]


def test_labels_no_longer_hardcode_a_unit():
    for f in FIELDS.values():
        assert "account ccy" not in f["label"] and "price units" not in f["label"]


def test_market_currencies():
    assert (MARKET_CURRENCY["US"], MARKET_CURRENCY["SG"], MARKET_CURRENCY["NSE"]) == ("USD", "SGD", "INR")
