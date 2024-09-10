import http.client
from sportmonks.src.fixture import *
from datetime import datetime

MY_TOKEN = 'MceRjLwnAieZF5k00xBaKAZo5iUio9W8enzx8rI3HaVChBRIdtGwhR6N3zzR'

def check_100_days(start_date,end_date):
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')

    date_difference = end_date_obj - start_date_obj

    return date_difference.days

if __name__ == "__main__":
    conn = http.client.HTTPSConnection("api.sportmonks.com")

    start_date = '2023-09-01'
    end_date = '2023-12-01'
    if check_100_days(start_date, end_date) > 100:
        raise ValueError(
            "The range of days in your request is too large. The maximum range is 100 days. Please narrow your range.")

    fixtures = get_Fixture_by_Date_Range(conn,MY_TOKEN,start_date,end_date)
    print(fixtures)