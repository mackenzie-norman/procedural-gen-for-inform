## Generate bus route
"""
Broadway & 17th is a room. Fourth Street is north of Broadway & 17th. Fifth Street is north of Fourth Street. Sixth Street is north of Fifth Street. Seventh Street is north of Sixth Street. Eighth Street is north of Seventh Street. Ninth Street is north of Eighth Street. Eleventh Street is north of Ninth Street. Fifteenth Street is north of Eleventh Street. Seventeenth Street is north of Fifteenth Street.

The COTA bus is a relatively-scheduled train-car in Broadway & 17th. The waiting duration of the COTA bus is 1 minute. The t-schedule of the COTA bus is the Table of Bus Schedule.

        Table of Bus Schedule
        transit time	destination
        1 minute	Broadway & 17th
        1 minute	Fourth Street
        1 minute	Fifth Street
        1 minute	Ninth Street
        3 minutes	Seventeenth Street
        3 minutes	Ninth Street
        1 minute	Fifth Street
        1 minute	Fourth Street
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

import networkx as nx
import requests
import pandas as pd


def get_stops_from_map(bus_number) -> list:
    # Optional: Set up Chrome options (e.g., headless mode)
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run in headless mode

    # Path to your ChromeDriver (make sure it's in PATH or provide full path)
    driver = webdriver.Chrome(options=options)

    time.sleep(3)
    url = f"https://ride.trimet.org/home/route/{bus_number}/"
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    elements = soup.find_all(
        lambda tag: tag.has_attr("class")
        and any(cls.startswith("styled__StopListBody") for cls in tag["class"])
    )

    element = elements[0]
    buttons = element.find_all("button")

    stops = [button.text.strip() for button in buttons]
    return stops


import networkx as nx
import requests
import pandas as pd


def get_df(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table")

    df = pd.read_html(str(table))
    return df[0]


def df_to_travel_times(df, stops):
    df.columns = [d.split(" Stop ID")[0] for d in df.columns]
    col_times = dict(
        zip(df.columns, [time.strptime("0" + l[:4], "%H:%M") for l in df.iloc[0]])
    )


def to_if7():
    BUS_NUMBER = 17
    stops = get_stops_from_map(BUS_NUMBER)
    stops = stops[20 : 20 + 17]
    stop1 = stops.pop(0)
    end_str = f"{stop1} is a room. "
    last_stop = stop1
    for s in stops:
        end_str += f"{s} is north of {last_stop}. "
        last_stop = s
    end_str += f"""\n\nThe {BUS_NUMBER} bus is a relatively-scheduled train-car in {stop1}. The waiting duration of the {BUS_NUMBER} bus is 1 minute. The t-schedule of the {BUS_NUMBER} bus is the Table of Bus Schedule.\n"""
    end_str += """
    Table of Bus Schedule
    transit time	destination
    """
    time_duration = 1
    for stop in [stop1] + stops:
        end_str += f"1 minute\t{stop}\n"

    print(end_str)


df = get_df("https://trimet.org/schedules/w/t1017_0.htm")
df_to_travel_times(df, None)
