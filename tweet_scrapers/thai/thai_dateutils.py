from dateutil import parser
import ast
import datetime


'''
This is made for this articles scraping only
3 components to handle
year (Buddhist calendar), month (use table below), date (thank fuck it's solar calendar)
'''


thai_months = {
    "ม.ค.": 1,
    "ก.พ.": 2,
    "มี.ค.": 3,
    "เม.ย.": 4,
    "พ.ค.": 5,
    "มิ.ย.": 6,
    "ก.ค.": 7,
    "ส.ค.": 8,
    "ก.ย.": 9,
    "ต.ค.": 10,
    "พ.ย.": 11,
    "ธ.ค.": 12,
}

year_offset = 543

def thai_date_convert(date_str):
    core_components = date_str.split(" ")[1:-1]
    # print(core_components)

    # first is date, keep
    date = core_components[0]
    month = thai_months[core_components[1]]
    year_gregorian = int(core_components[2]) - year_offset
    time_str = core_components[3]
    time_lst = time_str.split(":")
    time_lst.append("0")
    # print(time_lst)

    time_complete = [int(year_gregorian), int(month), int(date)]
    time_complete.extend([int(i) for i in time_lst])
    # print(time_complete)
    results = datetime.datetime(time_complete[0], time_complete[1], time_complete[2], time_complete[3], time_complete[4],
                                time_complete[5])
    # print(results)
    return results
