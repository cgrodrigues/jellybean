# Copyright 2026 Carlos Rodrigues
# SPDX-License-Identifier: Apache-2.0

"""Short README.md texts written into each dataset folder with ``--with-readme``.

They describe column meanings only (no hints about planted anomalies), to test
runs with and without human-provided context.
"""

DATASET_READMES: dict[str, str] = {
    "traffic": """# traffic

Call detail records of a mobile operator, one row per call, partitioned by day.

| column | meaning |
| --- | --- |
| call_id | unique identifier of the call |
| msisdn | phone number of the subscriber, international format (+351...) |
| destination_number | number on the other side of the call |
| destination_country | ISO country code of the destination (PT = domestic) |
| start_time | local start time of the call |
| duration_sec | duration of the call in seconds |
| charge | amount charged in EUR (0 for incoming calls) |
| cell_id | radio cell that served the call |
| imei | identifier of the handset used |
| call_type | outgoing or incoming |
""",
    "cust_segmentation": """# cust_segmentation

Customer master data, one row per customer.

| column | meaning |
| --- | --- |
| customer_id | subscriber phone number, national format (9 digits) |
| segment | commercial segment (Consumer, Business, Youth, Premium) |
| tariff_plan | current tariff plan |
| activation_date | date the line was activated |
| region | home region |
| credit_class | internal credit rating (A best, D worst) |
| age_band | age band of the subscriber |
| region_code | internal numeric code of the sales region |
""",
    "orders": """# orders

Online and in-store orders of the operator's shop, one row per order.

| column | meaning |
| --- | --- |
| order_id | unique identifier of the order |
| customer_id | subscriber phone number, national format (9 digits) |
| order_ts | timestamp of the order |
| product | product category |
| amount | order amount in EUR |
| channel | sales channel |
| payment_method | payment method |
| shipping_address_id | identifier of the shipping address |
| device_id | identifier of the device used to place the order |
""",
}


# --------------------------------------------------------------------------- #
# Top-up ledger (requirements_2.md section 9)                                 #
# --------------------------------------------------------------------------- #
_MOVEMENT_MEANINGS = {
    "RCH_VOUCHER": "recharge with a voucher (PIN); credit to MAIN",
    "RCH_DEALER": "electronic recharge at a dealer; credit to MAIN",
    "RCH_ATM": "recharge at an ATM (Multibanco); credit to MAIN",
    "RCH_APP": "recharge by card in the app or online; credit to MAIN",
    "BONUS_RCH": "bonus granted for a recharge (ref_movement_id = the recharge); credit to BONUS",
    "PROMO_CREDIT": "promotional credit without a recharge; credit to BONUS",
    "ADJ_CREDIT": "manual credit adjustment by an operator",
    "ADJ_DEBIT": "manual debit adjustment by an operator",
    "REVERSAL": "reversal of a recharge (ref_movement_id = the recharge); debit to MAIN",
    "BONUS_CLAWBACK": "cancellation of the bonus of a reversed recharge (ref_movement_id = the bonus); debit to BONUS",
    "P2P_OUT": "balance transfer sent to counterparty_msisdn; debit to MAIN",
    "P2P_IN": "balance transfer received from counterparty_msisdn; credit to MAIN",
    "ADV_GRANT": "balance advance; credit to MAIN",
    "ADV_RECOVERY": "recovery of an advance (ref_movement_id = the advance); debit to MAIN",
    "BUNDLE_PURCHASE": "purchase of a bundle; debit to MAIN or BONUS",
    "EXPIRY": "expiry of promotional balance; debit to BONUS",
}


def topup_readmes(codes: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """README.md texts for the top-up datasets.

    Args:
        codes: The opaque code mapping (``--opaque-codes``), so the dictionary
            lists the codes that are actually in the data.
    """
    codes = codes or {}
    movement = codes.get("movement_type", {})
    channel = codes.get("channel", {})
    types = "\n".join(f"| {movement.get(k, k)} | {v} |" for k, v in _MOVEMENT_MEANINGS.items())
    channels = ", ".join(
        f"{channel.get(k, k)} ({k.lower()})" if channel else k
        for k in ("VOUCHER", "POS", "ATM", "APP", "USSD", "BACKOFFICE", "SYSTEM")
    )
    return {
        "topup": f"""# topup

Ledger of prepaid wallet movements, one row per movement, partitioned by day.

Sign convention: `amount` is signed, positive for a credit and negative for a
debit. `balance_after = balance_before + amount` on every row. Each customer has
two wallets: MAIN (paid balance) and BONUS (promotional balance). Consumption
(calls, data) is not in this dataset, so balances do not carry over from one
movement to the next. Columns specific to a movement type are empty for the
other types.

| column | meaning |
| --- | --- |
| movement_id | unique identifier of the movement |
| msisdn | customer phone number, international format (+351...) |
| event_time | date and time of the movement |
| movement_type | type of movement (see below) |
| channel | {channels} |
| wallet | MAIN or BONUS |
| amount | signed value in EUR |
| face_value | value paid by the customer (paid recharges only) |
| balance_before / balance_after | wallet balance before / after the movement |
| voucher_serial, voucher_batch_id | voucher and its batch (voucher recharges only) |
| dealer_id | dealer (dealer recharges only) |
| agent_user | operator who made the adjustment (adjustments only) |
| promo_id | promotion (bonuses and promotional credits only) |
| reason_code | reason (adjustments and reversals) |
| ref_movement_id | movement this one refers to (a movement_id of this dataset) |
| counterparty_msisdn | the other customer (transfers only) |
| payment_ref | payment reference (electronic paid recharges only) |

| movement_type | meaning |
| --- | --- |
{types}
""",
        "promotions": """# promotions

Promotions, one row per promotion. A recharge bonus is `bonus_pct` % of the
recharge, at most `max_bonus`, for recharges of at least `min_recharge` between
`start_date` and `end_date`, at most `max_per_customer` times per customer.
Promotional credits (no recharge) have no `bonus_pct`: they credit `max_bonus`.
""",
        "dealers": """# dealers

Dealers that sell electronic recharges, one row per dealer. `dealer_type` is
INDEPENDENT (small shop), CHAIN (large retail chain) or ONLINE. `closing_hour` is
the hour the shop closes (24 = always open).
""",
        "voucher_batches": """# voucher_batches

Batches of recharge vouchers, one row per batch: face value, number of vouchers,
the date the batch was distributed to the selling dealer (`dealer_id`). Vouchers
can be redeemed after distribution.
""",
    }
