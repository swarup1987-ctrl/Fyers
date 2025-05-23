# holidays.py
from datetime import date

NSE_HOLIDAYS = {
    2025: {
        date(2025, 1, 26),   # Republic Day
        date(2025, 3, 31),   # Holi
        date(2025, 4, 10),   # Good Friday
        date(2025, 5, 1),    # Maharashtra Day
        date(2025, 8, 15),   # Independence Day
        date(2025, 10, 2),   # Gandhi Jayanti
        date(2025, 10, 24),  # Diwali (tentative)
        date(2025, 12, 25),  # Christmas
    },
    # Add previous/future years as needed
}

def is_trading_day(check_date):
    """Check if a date is a trading day (not weekend or holiday)"""
    if check_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    return check_date not in NSE_HOLIDAYS.get(check_date.year, set())
