"""Plain-English names for columns, category values, algorithms and models.

Use these on every chart and report so the slides never show raw codes like
`customer_type`, `Non Refund` or `lgbm`.
"""

from __future__ import annotations

FEATURE_NAMES = {
    # numeric
    "lead_time": "Lead time (days booked ahead)",
    "arrival_date_week_number": "Arrival week of the year",
    "arrival_date_day_of_month": "Arrival day of the month",
    "arrival_month_num": "Arrival month",
    "stays_in_weekend_nights": "Weekend nights",
    "stays_in_week_nights": "Weeknights",
    "total_nights": "Total nights",
    "adults": "Adults",
    "children": "Children",
    "babies": "Babies",
    "total_guests": "Total guests",
    "is_repeated_guest": "Repeat guest",
    "previous_cancellations": "Earlier bookings canceled",
    "previous_bookings_not_canceled": "Earlier bookings kept",
    "booking_changes": "Changes to the booking",
    "days_in_waiting_list": "Days on the waiting list",
    "adr": "Price per night (ADR)",
    "required_car_parking_spaces": "Parking spaces requested",
    "total_of_special_requests": "Special requests",
    "has_agent": "Booked through an agent",
    "has_company": "Booked by a company",
    # categorical
    "hotel": "Hotel (city or resort)",
    "meal": "Meal plan",
    "country": "Guest's country",
    "market_segment": "Market segment",
    "distribution_channel": "Booking channel",
    "reserved_room_type": "Room type reserved",
    "deposit_type": "Deposit type",
    "customer_type": "Customer type",
    "arrival_weekday": "Arrival weekday",
}

VALUE_NAMES = {
    "deposit_type": {
        "No Deposit": "No deposit", "Non Refund": "Non-refundable", "Refundable": "Refundable",
    },
    "market_segment": {
        "Online TA": "Online travel agent",
        "Offline TA/TO": "Offline agent / tour operator",
        "Complementary": "Complimentary (free stay)",
        "Aviation": "Aviation (airlines)",
    },
    "distribution_channel": {
        "TA/TO": "Travel agent / tour operator",
        "GDS": "Global distribution system (GDS)",
    },
    "customer_type": {
        "Transient": "Individual",
        "Transient-Party": "Individual, linked to another booking",
    },
    "reservation_status": {"Check-Out": "Checked out", "No-Show": "No-show"},
}

COUNTRY_NAMES = {
    "PRT": "Portugal", "GBR": "United Kingdom", "FRA": "France", "ESP": "Spain",
    "DEU": "Germany", "ITA": "Italy", "IRL": "Ireland", "BEL": "Belgium", "BRA": "Brazil",
    "NLD": "Netherlands", "USA": "United States", "CHE": "Switzerland", "CN": "China",
    "CHN": "China", "AUT": "Austria", "SWE": "Sweden", "POL": "Poland", "ISR": "Israel",
    "RUS": "Russia", "NOR": "Norway", "ROU": "Romania", "FIN": "Finland", "DNK": "Denmark",
    "AUS": "Australia", "AGO": "Angola", "LUX": "Luxembourg", "MAR": "Morocco",
    "TUR": "Turkey", "HUN": "Hungary", "ARG": "Argentina", "JPN": "Japan", "CZE": "Czechia",
    "IND": "India", "KOR": "South Korea", "GRC": "Greece", "DZA": "Algeria", "MEX": "Mexico",
    "CAN": "Canada",
}

# FLAML estimator codes -> algorithm family
ALGORITHM_NAMES = {
    "lgbm": "LightGBM (gradient boosting)",
    "xgboost": "XGBoost (gradient boosting)",
    "xgb_limitdepth": "XGBoost, depth-limited",
    "rf": "Random forest",
    "extra_tree": "Extra trees",
    "lrl1": "Logistic regression (L1)",
    "lrl2": "Logistic regression (L2)",
    "catboost": "CatBoost (gradient boosting)",
    "kneighbor": "k-nearest neighbors",
}

# Baseline model keys (src/baseline.py) -> display name
MODEL_NAMES = {
    "dummy_prior": "Dummy (always guesses the average)",
    "logistic_regression": "Logistic regression",
}


def value_name(col: str | None, value) -> str:
    return VALUE_NAMES.get(col, {}).get(value, str(value))


def feature_name(col: str) -> str:
    return FEATURE_NAMES.get(col, col.replace("_", " ").capitalize())


def algorithm_name(code: str) -> str:
    return ALGORITHM_NAMES.get(code, code)


def lower_first(text: str) -> str:
    """'Price per night (ADR)' -> 'price per night (ADR)' (keeps acronyms)."""
    return text[:1].lower() + text[1:]


def minus(text: str) -> str:
    """Use a real minus sign in numbers shown on slides."""
    return text.replace("-", "−")


def join_words(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
