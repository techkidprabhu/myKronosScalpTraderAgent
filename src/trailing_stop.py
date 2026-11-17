"""
Pure trailing-stop math - no broker calls in here on purpose, so it's
easy to unit test and identical across every agent in this project.
The orchestrator wires this up to real prices and real broker order
calls; this module just answers "given where price is now, what should
the stop be, and has it been hit?"

Trailing logic (Chandelier-exit style, works for both BUY and SELL):
  - Track the best price seen since entry (high_water for BUY, i.e.
    the highest close; low_water for SELL, the lowest close).
  - Trailing only ENGAGES once price has moved `trigger_points` in the
    favorable direction from entry - this avoids ratcheting a stop up
    on the first tiny wiggle right after entry.
  - Once engaged, the stop trails `trail_distance` behind the high/low
    -water mark. It only ever moves in the favorable direction - it
    never loosens, even if price pulls back without a new extreme.
  - If ATR is supplied and use_atr is set, trail_distance is computed
    as atr * atr_multiplier each update instead of a fixed distance,
    so the trail adapts to current volatility.
"""
from dataclasses import dataclass


@dataclass
class TrailConfig:
    trigger_points: float          # profit distance before trailing engages
    trail_distance: float          # fixed distance stop trails behind the extreme
    use_atr: bool = False
    atr_multiplier: float = 3.0    # trail_distance = atr * atr_multiplier when use_atr
    # Percent versions (of the entry / of the best price): when set they
    # replace trigger_points / trail_distance, so a $20 and a $500 stock trail
    # alike. ATR (use_atr with an ATR available) still wins for the distance.
    trigger_pct: float | None = None
    trail_pct: float | None = None


@dataclass
class TrailUpdate:
    new_stop: float
    new_extreme: float             # updated high_water (BUY) / low_water (SELL)
    trail_active: bool
    stop_changed: bool
    exit_triggered: bool           # True if current_price has crossed the stop
    target_hit: bool               # True if current_price has reached the target


def compute_update(
    direction: str,                # "BUY" or "SELL"
    entry_price: float,
    current_price: float,
    current_stop: float,
    current_extreme: float,        # position's stored high_water/low_water
    trail_active: bool,
    target: float | None,
    config: TrailConfig,
    atr: float | None = None,
) -> TrailUpdate:
    is_buy = direction == "BUY"

    # 1) update the running best price
    if is_buy:
        new_extreme = max(current_extreme, current_price)
        favorable_move = new_extreme - entry_price
    else:
        new_extreme = min(current_extreme, current_price)
        favorable_move = entry_price - new_extreme

    if config.use_atr and atr:
        trail_distance = atr * config.atr_multiplier
    elif config.trail_pct:
        trail_distance = new_extreme * config.trail_pct / 100
    else:
        trail_distance = config.trail_distance
    trigger = entry_price * config.trigger_pct / 100 if config.trigger_pct else config.trigger_points

    # 2) engage trailing once the trigger distance has been covered
    new_trail_active = trail_active or (favorable_move >= trigger)

    # 3) compute the candidate stop and only ratchet in the favorable direction
    new_stop = current_stop
    stop_changed = False
    if new_trail_active:
        candidate = (new_extreme - trail_distance) if is_buy else (new_extreme + trail_distance)
        if is_buy and candidate > current_stop:
            new_stop = candidate
            stop_changed = True
        elif not is_buy and candidate < current_stop:
            new_stop = candidate
            stop_changed = True

    # 4) check exit conditions against the (possibly just-updated) stop/target
    if is_buy:
        exit_triggered = current_price <= new_stop
        target_hit = target is not None and current_price >= target
    else:
        exit_triggered = current_price >= new_stop
        target_hit = target is not None and current_price <= target

    return TrailUpdate(
        new_stop=new_stop,
        new_extreme=new_extreme,
        trail_active=new_trail_active,
        stop_changed=stop_changed,
        exit_triggered=exit_triggered,
        target_hit=target_hit,
    )
