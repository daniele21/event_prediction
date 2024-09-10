def get_Fixture_by_Date_Range(conn,MY_TOKEN,start_date,end_date):
    payload = ''
    headers = {}

    conn.request("GET", f"/v3/football/fixtures/between/{start_date}/{end_date}?api_token={MY_TOKEN}", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")