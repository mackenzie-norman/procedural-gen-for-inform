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


def get_stops_from_map(bus_number):
    # Optional: Set up Chrome options (e.g., headless mode)
    options = Options()
    # options.add_argument("--headless")  # Uncomment to run in headless mode

    # Path to your ChromeDriver (make sure it's in PATH or provide full path)
    driver = webdriver.Chrome(options=options)

    time.sleep(3)
    url = f"https://ride.trimet.org/home/route/{bus_number}/"
    driver.get(url)


BUS_NUMBER = 17
# Load your HTML content (can be from a file or a string)

soup = BeautifulSoup(driver.page_source, "html.parser")

elements = soup.find_all(
    lambda tag: tag.has_attr("class")
    and any(cls.startswith("styled__StopListBody") for cls in tag["class"])
)

element = elements[0]
# Find the element with the specified class
# element = soup.select_one(".styled__StopListBody-sc-2xhfkr-0")

print(element)
# Find all <button> tags
buttons = element.find_all("button")

# Print them or extract attributes/text
stops = [button.text.strip() for button in buttons]
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
